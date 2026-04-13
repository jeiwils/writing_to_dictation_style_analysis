"""Focused writing-to-dictation analysis pipeline package."""

from importlib import import_module
from typing import Any

__all__ = ["run_dictation_pipeline", "generate_all_dictation_visualisations"]


def __getattr__(name: str) -> Any:
    """Lazy exports to avoid importing heavy submodules at package import time."""
    if name == "run_dictation_pipeline":
        return import_module(".e_orchestrator", __name__).run_dictation_pipeline
    if name == "generate_all_dictation_visualisations":
        return import_module(".f_text_dictation_visualisations", __name__).generate_all_dictation_visualisations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
