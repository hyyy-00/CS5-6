"""Atomic attempt status; superseded outputs are archived before reuse."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), 'utf-8')
    temporary.replace(path)


@contextmanager
def recorded_attempt(directory, filename, *, metadata=None, outputs=()):
    """Publish passed only after the entire guarded operation completes.

    Callers creating immutable runs must reject an existing directory first.
    Mutable standalone outputs are preserved in previous_attempts/ before reuse.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex
    names = list(dict.fromkeys([filename, *outputs]))
    for name in names:
        if Path(name).name != name:
            raise ValueError('Attempt output names must be local filenames')
    existing = [directory/name for name in names if (directory/name).is_file()]
    if existing:
        archive = directory/'previous_attempts'/attempt
        archive.mkdir(parents=True, exist_ok=False)
        for source in existing:
            if source.is_symlink() or not source.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Attempt output cannot link outside its directory')
            source.replace(archive/source.name)
    base = {**(metadata or {}), 'attempt_id': attempt, 'started_utc': timestamp()}
    state = {**base, 'status': 'running'}
    write_status(directory/filename, state)
    try:
        yield state
        state.update(status='passed', finished_utc=timestamp())
        write_status(directory/filename, state)
    except BaseException as error:
        failed = {**base, 'status': 'failed', 'finished_utc': timestamp(),
                  'stage': state.get('stage', 'execution'),
                  'error_type': type(error).__name__, 'error': str(error),
                  'outputs_valid': False}
        write_status(directory/filename, failed)
        raise
