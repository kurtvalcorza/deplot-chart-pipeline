"""The standalone notebook runs every embedded module in one namespace, so a top-level name bound by two
modules with different definitions silently replaces the earlier one (a later module's helper would be
called by the earlier module's code). The package's own imports keep modules apart, so only this check
catches it offline."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bindings(source: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names[target.id] = ast.dump(node)
    return names


def test_embedded_modules_do_not_rebind_each_others_top_level_names() -> None:
    notebooks = sorted((ROOT / "tutorials").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        seen: dict[str, tuple[str, str]] = {}
        clashes = []
        for cell in cells:
            module = cell.get("metadata", {}).get("dimer", {}).get("embedded_module")
            if not module:
                continue
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for name, definition in _bindings(source).items():
                if name in seen and seen[name][1] != definition:
                    clashes.append(f"{name}: {seen[name][0]} and {module}")
                seen[name] = (module, definition)
        assert not clashes, f"{path.name}: embedded modules rebind {clashes}"
