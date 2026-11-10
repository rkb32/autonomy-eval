"""SystemID flight in ACRO: no angle loop, so the measured margin is of the rate loop alone.

ACRO has manual throttle, so SystemID's axis check passes, and its roll target is the pilot's rate
(zero here), with the chirp added on top. See scripts/sitl_sysid.py for the flight sequence.
"""
import sys

import sitl_sysid

ACRO = 1

if __name__ == "__main__":
    sys.exit(sitl_sysid.run(ACRO))
