"""How uploaded flight logs are handled: where they may live, and for how long.

Two rules, both enforced here rather than left to the caller:

* A reader that needs a file path gets a file in a temporary directory that is deleted when the
  call returns, including when the reader raises. Nothing else is written to disk.
* A reader's result is remembered only inside one user's session. The memo is passed a mapping
  owned by that session (st.session_state in the app), so one user's results are never served to
  another, and the memo is bounded and can be wiped when the files are removed.

These rules are about the app's own handling. They do not stop the Streamlit host from seeing an
upload while it is in memory, and they do not cover Streamlit's own copy of the uploaded file.
"""
from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

MEMO_KEY = "_user_log_memo"
# One upload with 3 .bin logs makes about 20 reader calls per rerun, so a small cap would re-parse on
# every click. Each entry is one session's derived result, freed when the session ends.
MEMO_MAX_ENTRIES = 64
# Set AUTONOMY_EVAL_HOSTED=1 on a server that anyone else can reach. Uploads are then turned off, so
# no visitor's file reaches that host. Local use leaves it unset.
HOSTED_ENV = "AUTONOMY_EVAL_HOSTED"


def uploads_enabled() -> bool:
    return os.environ.get(HOSTED_ENV) != "1"


def read_in_temp(name: str, data: bytes, reader: Callable[..., Any], *args: Any) -> Any:
    """Run `reader(path, *args)` on the uploaded bytes, written to a temporary directory that is
    removed afterwards, even if the reader raises."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        return reader(path, *args)


def memo(store: MutableMapping, reader: Callable[..., Any], name: str, data: bytes, *args: Any) -> Any:
    """Like read_in_temp, but the result is kept in `store` so the next call in the same session is
    free. `store` must be one session's own mapping; never a module-level or Streamlit cache, which
    would be shared between users. Keys include a full SHA-256 of the bytes, so a different file
    with the same name is never served a stale result."""
    key = (getattr(reader, "__module__", ""), getattr(reader, "__qualname__", repr(reader)),
           name, hashlib.sha256(data).hexdigest(), args)
    cache: OrderedDict = store.setdefault(MEMO_KEY, OrderedDict())
    # Callers get a copy, so changing a returned result cannot change what the next call sees.
    if key in cache:
        cache.move_to_end(key)
        return _handout(cache[key])
    value = read_in_temp(name, data, reader, *args)
    cache[key] = value
    while len(cache) > MEMO_MAX_ENTRIES:
        cache.popitem(last=False)
    return _handout(value)


def _handout(value: Any) -> Any:
    """What a caller receives. Containers are rebuilt so the caller cannot change the stored result.
    Arrays are not copied (that was slow on every rerun); each caller gets a read-only view instead."""
    if isinstance(value, np.ndarray):
        view = value.view()
        view.flags.writeable = False
        return view
    if isinstance(value, dict):
        return {k: _handout(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_handout(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_handout(v) for v in value)
    return copy.deepcopy(value)


def wipe(store: MutableMapping) -> None:
    """Forget every remembered result in this session, e.g. when the uploaded files are removed."""
    store.pop(MEMO_KEY, None)
