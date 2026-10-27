"""Read-only, standard-library check of the delivered file inventory.

This checks bytes against RELEASE_MANIFEST.json. It does not run model code,
verify numerical results, or authenticate the manifest's claimed provenance.
Unlisted files (for example, new runs/) are intentionally permitted.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


MANIFEST_NAME = "RELEASE_MANIFEST.json"
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def unique_object(pairs):
    """Reject duplicate JSON keys rather than silently replacing an entry."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def safe_relative_name(name):
    """Require one canonical, portable, relative POSIX file path."""
    return (
        isinstance(name, str)
        and bool(name)
        and "\\" not in name
        and not any(ord(character) < 32 for character in name)
        and not PurePosixPath(name).is_absolute()
        and not PureWindowsPath(name).drive
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def valid_sha256(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def check_release(root):
    """Return JSON-ready status; read listed regular files without changing them."""
    errors = []
    checked_files = 0

    def error(code, message, path=None):
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = path
        errors.append(item)

    def result():
        return {
            "status": "failed" if errors else "passed",
            "passed": not errors,
            "checked_files": checked_files,
            "errors": errors,
        }

    try:
        root = Path(root).resolve(strict=True)
        manifest_path = root / MANIFEST_NAME
        if manifest_path.is_symlink():
            error("unsafe_path", "The release manifest must not be a symlink.", MANIFEST_NAME)
            return result()
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, RuntimeError) as exc:
        error("manifest_unreadable", str(exc), MANIFEST_NAME)
        return result()
    except (ValueError, UnicodeError) as exc:
        error("invalid_manifest", str(exc), MANIFEST_NAME)
        return result()

    if not isinstance(manifest, dict):
        error("invalid_manifest", "Manifest must be a JSON object.")
        return result()
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        error("invalid_manifest", "schema_version must be integer 1.")
    archive = manifest.get("source_archive")
    if (
        not isinstance(archive, dict)
        or not safe_relative_name(archive.get("name"))
        or "/" in archive["name"]
        or not valid_sha256(archive.get("sha256"))
    ):
        error("invalid_manifest", "source_archive requires a basename and SHA-256 hash.")
    scope = manifest.get("verification_scope")
    if not isinstance(scope, str) or not scope.strip():
        error("invalid_manifest", "verification_scope must be a non-empty string.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        error("invalid_manifest", "files must be a non-empty object.")
        return result()

    for name, metadata in files.items():
        if not safe_relative_name(name) or name.casefold() == MANIFEST_NAME.casefold():
            error("unsafe_path", "Use a relative file path; exclude the manifest itself.", name)
            continue
        if (
            not isinstance(metadata, dict)
            or not valid_sha256(metadata.get("sha256"))
            or type(metadata.get("bytes")) is not int
            or metadata["bytes"] < 0
        ):
            error("invalid_manifest", "File metadata requires SHA-256 and nonnegative bytes.", name)
            continue
        try:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path == manifest_path:
                error("unsafe_path", "Resolved file path escapes the root or targets the manifest.", name)
                continue
            if not path.exists():
                error("missing_file", "Listed file is missing.", name)
                continue
            if not path.is_file():
                error("not_regular_file", "Listed path is not a regular file.", name)
                continue
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            checked_files += 1
            if size != metadata["bytes"]:
                error("size_mismatch", f"Expected {metadata['bytes']} bytes; found {size}.", name)
            if digest.hexdigest() != metadata["sha256"].lower():
                error("sha256_mismatch", "SHA-256 differs from the release manifest.", name)
        except (OSError, RuntimeError, ValueError) as exc:
            error("file_unreadable", str(exc), name)
    return result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Project root containing RELEASE_MANIFEST.json (default: script's project root).",
    )
    status = check_release(parser.parse_args().root)
    print(json.dumps(status, ensure_ascii=True, indent=2))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
