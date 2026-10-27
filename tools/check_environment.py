"""Check dependencies and the processed-data training/replay inputs."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.contracts import environment, source_identity
from pipeline.inputs import preflight
from pipeline.learning import implementations
from pipeline.trading import implementation
env = environment()
paths, missing = preflight()
paths += [ROOT / 'external_validation/ensemble_ranker/inputs/test40.csv',
          ROOT / 'external_validation/shared_inputs/inputs/test40_with_prices.csv']
missing += [str(p) for p in paths if not p.is_file()]
if missing:
    raise SystemExit('Missing processed inputs: ' + repr(missing))
modules = [implementations(), implementation()]
assert all(Path(m.__file__).resolve().is_relative_to(ROOT) for m in modules)
print(json.dumps(dict(status='passed', environment=env,
    processed_input_files=[p.relative_to(ROOT).as_posix() for p in paths],
    tracked_runtime_files=len(source_identity())), indent=2))
