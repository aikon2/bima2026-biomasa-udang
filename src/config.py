"""Repo-root-relative configuration and path helpers.

All paths are resolved against the repository root (derived from this file's
location) so scripts behave the same regardless of the current working directory.
"""
from pathlib import Path
from typing import Any, Dict, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """Return an absolute path; relative inputs are anchored at REPO_ROOT."""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_yaml(relative_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML mapping from a repo-relative path (empty file -> {})."""
    yaml_path = resolve_path(relative_path)
    try:
        with open(str(yaml_path), "r") as yaml_file:
            loaded = yaml.safe_load(yaml_file)
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError("Failed to load config {}: {}".format(yaml_path, error))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RuntimeError("Config {} must contain a mapping".format(yaml_path))
    return loaded
