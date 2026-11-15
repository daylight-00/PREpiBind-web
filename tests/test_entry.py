"""The entry scripts: the first session of a process must draw its page once (see the comment in app.py). The behaviour itself needs a browser; this keeps the two things it rests on from being edited away."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def switches_legacy_mode_off(tree):
    """A module-level `<something>.uses_pages_directory = False`."""
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "uses_pages_directory" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in tree.body
    )


def test_app_py_switches_the_legacy_pages_mode_off_and_builds_the_navigation_only_as_the_main_script():
    tree = ast.parse((ROOT / "app.py").read_text())
    assert switches_legacy_mode_off(tree)
    guarded = [node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) and "st.navigation" in ast.unparse(node)]
    outside = [node for node in tree.body if not isinstance(node, ast.If) and "navigation" in ast.unparse(node)]
    assert len(guarded) == 1 and not outside


def test_serve_py_switches_the_legacy_pages_mode_off_before_the_first_request():
    assert switches_legacy_mode_off(ast.parse((ROOT / "serve.py").read_text()))
