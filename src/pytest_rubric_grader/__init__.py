"""pytest-rubric-grader: rubric-based LLM-as-judge testing for pytest."""

# Lazy imports to avoid circular dependency between plugin.py and calibration.py.
# calibration.py uses GraderLLM from plugin.py under TYPE_CHECKING only;
# importing both at module level here could break if that changes.


def __getattr__(name: str):  # noqa: ANN001
    if name in ("CalibrationResult", "calibrate"):
        from pytest_rubric_grader.calibration import CalibrationResult, calibrate

        _exports = {"CalibrationResult": CalibrationResult, "calibrate": calibrate}
        return _exports[name]
    if name in ("GraderLLM", "OpenAICompatibleGrader"):
        from pytest_rubric_grader.plugin import GraderLLM, OpenAICompatibleGrader

        _exports = {"GraderLLM": GraderLLM, "OpenAICompatibleGrader": OpenAICompatibleGrader}
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CalibrationResult",
    "GraderLLM",
    "OpenAICompatibleGrader",
    "calibrate",
]
