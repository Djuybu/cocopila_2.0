"""Utilities; optional GPU dependencies are not imported eagerly."""
__all__ = ["SubmissionValidator", "evaluate_submission", "VRAMMonitor"]


def __getattr__(name):
    if name == "SubmissionValidator":
        from src.submission.validator import SubmissionValidator
        return SubmissionValidator
    if name == "evaluate_submission":
        from src.evaluation.evaluator import evaluate_submission
        return evaluate_submission
    if name == "VRAMMonitor":
        from src.utils.vram_monitor import VRAMMonitor
        return VRAMMonitor
    raise AttributeError(name)
