"""Fingerprint the preprocessing definition and number of input observations."""

import ast
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path


class _RemoveDocstrings(ast.NodeTransformer):
    """Remove leading docstrings; comments are already absent from the AST."""

    def _visit_body(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body


def preprocessing_source_paths() -> tuple[Path, ...]:
    """List all four definition files so moved calculations remain covered."""
    directory = Path(__file__).parent
    return tuple(directory / name for name in (
        "preprocess_mojave.py",
        "mojave_metadata.py",
        "dataset_summary.py",
        "preprocessing_hash.py",
    ))


def calculate_preprocessing_hash(
    dataset_length: int,
    source_paths: Sequence[Path] | None = None,
) -> str:
    """Hash code without comments/docstrings, followed by input dataset length.

    Formatting is ignored too. Length includes observations that later fail.
    Source paths can be supplied explicitly for another preprocessing definition.
    """
    if isinstance(dataset_length, bool) or not isinstance(dataset_length, int):
        raise ValueError("Dataset length must be an integer.")
    if dataset_length < 0:
        raise ValueError("Dataset length must not be negative.")
    paths = preprocessing_source_paths() if source_paths is None else source_paths
    definitions = []
    for path in sorted(map(Path, paths)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree = _RemoveDocstrings().visit(tree)
        definitions.append(ast.dump(tree, include_attributes=False))
    if not definitions:
        raise ValueError("At least one preprocessing source file is required.")
    payload = json.dumps([definitions, dataset_length], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_hash_to_current_hash(
    summary_path: str | Path,
    dataset_length: int,
    source_paths: Sequence[Path] | None = None,
    *,
    preprocessing_settings: dict | None = None,
) -> bool:
    """Check a saved summary against the current definition and input length.

    Missing, malformed or unfinished summaries are not current. Length counts
    input observations, including failures, rather than successfully saved crops.
    Supplied settings are compared with the corresponding saved settings.
    Errors reading the preprocessing source files propagate to the caller.
    """
    current_hash = calculate_preprocessing_hash(dataset_length, source_paths)
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    saved_settings = summary.get("preprocessing_settings")
    settings_match = preprocessing_settings is None or (
        isinstance(saved_settings, dict)
        and all(saved_settings.get(name) == value for name, value in preprocessing_settings.items())
    )
    return (
        summary.get("preprocessing_finished") is True
        and type(summary.get("dataset_length")) is int
        and summary["dataset_length"] == dataset_length
        and summary.get("preprocessing_hash") == current_hash
        and settings_match
    )
