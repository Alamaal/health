"""
Tests for TeamClassifier empty-crop guards and the main() display flag.

These tests target the fixes from the problem statement:
- TeamClassifier.extract_features() / fit() should raise clearly on empty crops
- run_team_classification / run_radar should raise RuntimeError when no crops
  were collected (not a silent np.concatenate failure)
- main() should not call cv2.imshow/destroyAllWindows unless display=True
"""

import ast
import os

import numpy as np
import pytest

# Path helpers
_REPO = os.path.join(os.path.dirname(__file__), "..")
_TEAM_PY = os.path.join(_REPO, "sports", "common", "team.py")
_MAIN_PY = os.path.join(_REPO, "examples", "soccer", "main.py")


def _parse(path):
    return ast.parse(open(path).read())


# ---------------------------------------------------------------------------
# TeamClassifier empty-crop guards (AST + functional stub)
# ---------------------------------------------------------------------------

class TestTeamPyEmptyCropsGuard:
    """Verify that team.py has the empty-crops guard at the source level."""

    def test_extract_features_has_empty_guard(self):
        """extract_features() must have an early-return for empty crops."""
        tree = _parse(_TEAM_PY)
        extract = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "extract_features"
        )
        # Look for: if len(crops) == 0: return ...
        source = open(_TEAM_PY).read()
        lines = source.splitlines()
        func_lines = lines[extract.lineno - 1: extract.end_lineno]
        guard_present = any(
            "len(crops) == 0" in line or "not crops" in line
            for line in func_lines
        )
        assert guard_present, (
            "extract_features() has no empty-crops guard (len(crops)==0 check)"
        )

    def test_fit_raises_on_empty_crops(self):
        """fit() must raise ValueError when crops is empty."""
        tree = _parse(_TEAM_PY)
        fit_func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "fit"
        )
        source = open(_TEAM_PY).read()
        lines = source.splitlines()
        func_lines = lines[fit_func.lineno - 1: fit_func.end_lineno]
        func_text = "\n".join(func_lines)
        # Must check for empty crops and raise ValueError
        has_empty_check = "len(crops) == 0" in func_text or "not crops" in func_text
        has_raise = "raise ValueError" in func_text
        assert has_empty_check, "fit() does not check for empty crops"
        assert has_raise, "fit() does not raise ValueError for empty crops"

    def test_extract_features_empty_returns_array_functionally(self):
        """extract_features([]) must return an empty ndarray without needing model."""
        # We test only the early-return branch by monkey-patching the class.
        # Import the module; if torch is unavailable, skip.
        pytest.importorskip("torch", reason="torch not installed in this environment")
        from sports.common.team import TeamClassifier  # noqa: PLC0415

        class _StubClassifier(TeamClassifier):
            def __init__(self):
                self.device = 'cpu'
                self.use_fp16 = False
                self.batch_size = 32
                # skip model loading

        clf = _StubClassifier()
        result = clf.extract_features([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_fit_empty_raises_value_error_functionally(self):
        """fit([]) must raise ValueError with the expected message."""
        pytest.importorskip("torch", reason="torch not installed in this environment")
        from sports.common.team import TeamClassifier  # noqa: PLC0415

        class _StubClassifier(TeamClassifier):
            def __init__(self):
                self.device = 'cpu'
                self.use_fp16 = False
                self.batch_size = 32

        clf = _StubClassifier()
        with pytest.raises(ValueError, match="requires at least one crop"):
            clf.fit([])


class TestMainPyEmptyCropsGuard:
    """Verify that main.py guards run_team_classification / run_radar against empty crops."""

    def test_run_team_classification_has_empty_crops_guard(self):
        source = open(_MAIN_PY).read()
        tree = ast.parse(source)
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_team_classification"
        )
        lines = source.splitlines()
        func_lines = lines[func.lineno - 1: func.end_lineno]
        func_text = "\n".join(func_lines)
        assert "not crops" in func_text or "len(crops) == 0" in func_text, (
            "run_team_classification() has no empty-crops guard before fit()"
        )
        assert "RuntimeError" in func_text, (
            "run_team_classification() does not raise RuntimeError for empty crops"
        )

    def test_run_radar_has_empty_crops_guard(self):
        source = open(_MAIN_PY).read()
        tree = ast.parse(source)
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_radar"
        )
        lines = source.splitlines()
        func_lines = lines[func.lineno - 1: func.end_lineno]
        func_text = "\n".join(func_lines)
        assert "not crops" in func_text or "len(crops) == 0" in func_text, (
            "run_radar() has no empty-crops guard before fit()"
        )
        assert "RuntimeError" in func_text, (
            "run_radar() does not raise RuntimeError for empty crops"
        )


# ---------------------------------------------------------------------------
# main() display flag
# ---------------------------------------------------------------------------

class TestMainDisplayFlag:
    """The display kwarg controls cv2.imshow usage."""

    def test_main_has_display_parameter(self):
        """main() function signature must include a 'display' parameter."""
        tree = _parse(_MAIN_PY)
        main_func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'main'
        )
        all_args = [arg.arg for arg in main_func.args.args]
        all_args += [arg.arg for arg in main_func.args.kwonlyargs]
        assert 'display' in all_args, (
            f"'display' parameter not found in main(). Found: {all_args}"
        )

    def test_main_display_defaults_to_false(self):
        """main() display parameter must default to False."""
        tree = _parse(_MAIN_PY)
        main_func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'main'
        )
        all_args = main_func.args.args
        defaults = main_func.args.defaults
        defaulted_args = all_args[len(all_args) - len(defaults):]
        default_map = {}
        for arg, val in zip(defaulted_args, defaults):
            try:
                default_map[arg.arg] = ast.literal_eval(val)
            except Exception:
                pass
        for arg, val in zip(main_func.args.kwonlyargs, main_func.args.kw_defaults):
            if val is not None:
                try:
                    default_map[arg.arg] = ast.literal_eval(val)
                except Exception:
                    pass
        assert 'display' in default_map, "No default found for 'display' in main()"
        assert default_map['display'] is False, (
            f"Expected display=False, got {default_map['display']!r}"
        )

    def test_argparse_has_display_flag(self):
        """CLI must expose a --display flag."""
        source = open(_MAIN_PY).read()
        assert "'--display'" in source or '"--display"' in source, (
            "--display argument not found in argparse setup"
        )

    def test_imshow_guarded_by_display(self):
        """cv2.imshow must only be called inside an 'if display:' block."""
        source = open(_MAIN_PY).read()
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == 'imshow'
                and isinstance(func.value, ast.Name)
                and func.value.id == 'cv2'
            ):
                continue
            imshow_lineno = node.lineno - 1  # 0-indexed
            found_guard = any(
                'if display' in lines[i]
                for i in range(max(0, imshow_lineno - 10), imshow_lineno)
            )
            assert found_guard, (
                f"cv2.imshow at line {node.lineno} is not guarded by 'if display'"
            )

    def test_no_unconditional_destroyallwindows(self):
        """cv2.destroyAllWindows must be guarded by display, not always called."""
        source = open(_MAIN_PY).read()
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == 'destroyAllWindows'
                and isinstance(func.value, ast.Name)
                and func.value.id == 'cv2'
            ):
                continue
            daw_lineno = node.lineno - 1
            found_guard = any(
                'if display' in lines[i]
                for i in range(max(0, daw_lineno - 10), daw_lineno)
            )
            assert found_guard, (
                f"cv2.destroyAllWindows at line {node.lineno} is not guarded by 'if display'"
            )

