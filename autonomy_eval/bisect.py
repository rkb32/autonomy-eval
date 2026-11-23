"""Point at the commits that could explain a flagged regression, instead of just naming the build.

Build labels are ArduPilot git short SHAs (see FlightSummary.label), so the commit range between
the last clean build and a flagged one is a real, fetchable thing -- this is how the cross-track
regression in the README was traced, by hand, to a GitHub compare link. `localize()` automates
that: it fetches the range from GitHub, keeps only commits that touch source directories plausibly
related to the metric that regressed, and (only if ANTHROPIC_API_KEY is set AND AUTONOMY_EVAL_SEND_TO_AI=1)
asks a model for a single "likely true/false positive" read grounded only in those commit messages
and files -- never asked to invent ArduPilot internals it wasn't given. The AI call sends the metric
name and commit data (never log contents), and it is off unless both settings are present.

This is a lead, not a diagnosis: a commit touching the right directory is a place to look, not
proof. The AI verdict is the same kind of lead, one level more digested.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .compare import Comparison
from .metrics import FlightSummary

REPO = "ArduPilot/ardupilot"
SHA_RE = re.compile(r"[0-9a-f]{7,40}")  # used with fullmatch(): a trailing newline must not pass
MAX_COMMITS = 25  # bound on per-commit API calls; wider ranges are reported but not localized
# The AI read sends the metric name and commit data to Anthropic. It needs this opt-in as well as the
# key, so having a key in the environment for another tool never sends anything.
AI_OPT_IN_ENV = "AUTONOMY_EVAL_SEND_TO_AI"

# Metric prefix -> source directories whose changes could plausibly explain it. Verified against
# the real ardupilot/libraries tree; a metric with no entry here isn't localized.
SUBSYSTEMS: dict[str, tuple[str, ...]] = {
    "xtrack": ("libraries/AC_WPNav", "libraries/AR_WPNav", "libraries/AC_AttitudeControl",
               "libraries/APM_Control", "libraries/AP_L1_Control", "libraries/AP_Navigation"),
    "pos_err": ("libraries/AC_WPNav", "libraries/AC_AttitudeControl", "libraries/AP_InertialNav"),
    "att_err": ("libraries/AC_AttitudeControl", "libraries/AP_Motors", "libraries/AP_InertialSensor"),
    "ekf": ("libraries/AP_NavEKF3", "libraries/AP_NavEKF2", "libraries/AP_NavEKF", "libraries/AP_AHRS",
            "libraries/AP_InertialNav"),
    "vibe": ("libraries/AP_InertialSensor",),
    "clip": ("libraries/AP_InertialSensor",),
    "gps": ("libraries/AP_GPS",),
    "hdop": ("libraries/AP_GPS",),
    "batt": ("libraries/AP_BattMonitor",),
}


@dataclass
class SuspectCommit:
    sha: str
    message: str
    files: list[str]


@dataclass
class AIVerdict:
    label: str      # "likely true positive" | "likely false positive" | "uncertain"
    reason: str


@dataclass
class Bisection:
    metric: str
    base: str
    head: str
    compare_url: str
    total_commits: int = 0
    suspects: list[SuspectCommit] = field(default_factory=list)
    ai: AIVerdict | None = None
    error: str | None = None


def _get(url: str) -> dict:
    # Only GitHub's API is ever called, so refuse any other address (for example a local file).
    if not url.startswith("https://api.github.com/"):
        raise ValueError("refusing to open a URL outside api.github.com")
    headers = {"User-Agent": "autonomy-eval", "Accept": "application/vnd.github+json"}
    # Unauthenticated calls are limited to 60 an hour; a token raises that to 5000. It is read from the
    # environment only, so it is never written to a file or a commit.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with _opener().open(req, timeout=10) as r:  # nosec B310: scheme and host checked above
        return json.load(r)


class _GitHubOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only if it stays on api.github.com. Otherwise the token could go to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://api.github.com/"):
            host = urllib.parse.urlparse(newurl).netloc or "an unknown place"
            raise urllib.error.URLError(f"refusing to follow a redirect to {host}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    return urllib.request.build_opener(_GitHubOnlyRedirects)


def _subsystem_dirs(metric: str) -> tuple[str, ...] | None:
    return next((dirs for prefix, dirs in SUBSYSTEMS.items() if metric.startswith(prefix)), None)


def localize(metric: str, base: str, head: str) -> Bisection | None:
    """Commits between `base` and `head` that touch directories relevant to `metric`. None if
    either label isn't a real git SHA (e.g. a filename-derived label) or the metric isn't mapped."""
    dirs = _subsystem_dirs(metric)
    if dirs is None or not (SHA_RE.fullmatch(base) and SHA_RE.fullmatch(head)):
        return None
    b = Bisection(metric, base, head, f"https://github.com/{REPO}/compare/{base}...{head}")
    try:
        cmp = _get(f"https://api.github.com/repos/{REPO}/compare/{base}...{head}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        b.error = f"couldn't reach GitHub: {e}"
        return b
    commits = cmp.get("commits", [])
    b.total_commits = len(commits)
    if len(commits) > MAX_COMMITS:
        b.error = f"{len(commits)} commits in range, too many to check individually"
        return b
    failed = 0
    for c in commits:
        try:
            detail = _get(f"https://api.github.com/repos/{REPO}/commits/{c['sha']}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            failed += 1          # a failed lookup must not look like "no suspects"
            continue
        files = [f["filename"] for f in detail.get("files", [])]
        if any(f.startswith(d) for f in files for d in dirs):
            b.suspects.append(SuspectCommit(c["sha"][:8], c["commit"]["message"].splitlines()[0],
                                             [f for f in files if any(f.startswith(d) for d in dirs)]))
    if failed:
        b.error = (f"{failed} of {len(commits)} commit lookups failed (GitHub rate limit is 60 per hour "
                   f"unauthenticated); the suspect list is incomplete")
    return b


SYSTEM = (
    "You judge whether a flight-metric regression is a real firmware bug or noise, given only the "
    "commits between the last clean build and the flagged one that touch a plausibly relevant source "
    "directory. Base your verdict only on what the commit messages and filenames say. A commit that "
    "renames things, moves parameter tables, touches build tooling, comments, or tests is not a "
    "behavior change. A commit whose message describes a change to gains, thresholds, control logic, "
    "filtering, or timing in the relevant code is. If there are no suspect commits at all, that itself "
    "is evidence of a false positive. Never invent ArduPilot internals you were not given.\n\n"
    'Respond with only a JSON object: {"verdict": "likely true positive" | "likely false positive" | '
    '"uncertain", "reason": "<one sentence, cite a commit message or the absence of one>"}'
)


def ai_enabled(api_key: str | None = None) -> bool:
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get(AI_OPT_IN_ENV) == "1"


def lookup(metric: str, base: str, head: str, store) -> Bisection | None:
    """localize(), remembered in `store` (one session's mapping) only when it succeeds. A failed or
    rate-limited lookup is not kept, so the next try asks GitHub again."""
    key = f"lookup|{metric}|{base}|{head}"
    if key in store:
        return store[key]
    b = localize(metric, base, head)
    if b is not None and not b.error:
        store[key] = b
    return b


def ai_verdict(b: Bisection, api_key: str | None = None) -> AIVerdict | None:
    """The AI read. If the caller passes a key, that key is used; otherwise ANTHROPIC_API_KEY is used
    (the command line and the app both call this without a key today). The key is never stored or
    printed. On failure only the error's type is reported, because an error's text can contain part
    of the request."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if b.error or not key or os.environ.get(AI_OPT_IN_ENV) != "1":
        return None
    import anthropic

    payload = {
        "metric": b.metric, "commits_in_range": b.total_commits,
        "suspect_commits": [{"sha": s.sha, "message": s.message, "files": s.files} for s in b.suspects],
    }
    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model="claude-sonnet-5", max_tokens=300, system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
        )
    except anthropic.APIError as e:
        print(f"warning: AI verdict failed ({type(e).__name__}), continuing without it", file=sys.stderr)
        return None
    text = "".join(part.text for part in response.content if part.type == "text").strip()
    try:
        data = json.loads(text)
        return AIVerdict(data["verdict"], data["reason"])
    except (json.JSONDecodeError, KeyError):
        return None


def neighbor_build(candidate: FlightSummary, ordered_runs: list[FlightSummary]) -> tuple[FlightSummary, FlightSummary] | None:
    """(base, head) for bisecting `candidate`: the build right before it in `ordered_runs` (oldest to
    newest) as base, so the range covers what changed *into* the regression. If `candidate` is the
    first build in the list, there's nothing earlier to diff -- fall back to the build right after it,
    which instead answers "did this regression persist, or did the very next build already differ in
    a way that could explain it going away" (this is how the bf080274 regression in the README was
    traced by hand: it's the oldest build in that history, so there was no earlier one to check)."""
    for i, r in enumerate(ordered_runs):
        if r is candidate:
            if i > 0:
                return ordered_runs[i - 1], candidate
            if i + 1 < len(ordered_runs):
                return candidate, ordered_runs[i + 1]
            return None
    return None


def for_comparison(c: Comparison, ordered_runs: list[FlightSummary]) -> Bisection | None:
    """Localize + AI-verdict the top regression in `c`, if there's a neighboring build to diff
    against and the metric maps to a known subsystem. This is the one entry point CLI/app code
    should use."""
    if not c.regressions:
        return None
    pair = neighbor_build(c.candidate, ordered_runs)
    if pair is None:
        return None
    base, head = pair
    b = localize(c.regressions[0].metric, base.label, head.label)
    if b is not None and not b.error:
        b.ai = ai_verdict(b)
    return b
