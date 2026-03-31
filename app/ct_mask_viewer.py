from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ct_viewer.ct_mask_viewer import *  # noqa: F401,F403


if __name__ == "__main__":
    from ct_viewer.ct_mask_viewer import main

    raise SystemExit(main())
