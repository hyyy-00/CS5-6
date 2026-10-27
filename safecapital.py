"""Default: load the delivered Ensemble models, predict, and backtest."""
from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if '--retrain-v1' in args:
        if '--legacy-ensemble' in args or '--saved-ensemble' in args:
            raise ValueError('Select one prediction mode')
        from pipeline.runner import main as pipeline_main
        return pipeline_main([arg for arg in args if arg != '--retrain-v1'])
    from saved_ensemble import main as saved_main
    return saved_main([arg for arg in args if arg not in ['--saved-ensemble', '--legacy-ensemble']])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, ModuleNotFoundError) as error:
        print(
            f"Required dependency unavailable: {error}. Install requirements-repro.txt in an isolated environment.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as error:
        print(f"Cannot start SafeCapital run: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
