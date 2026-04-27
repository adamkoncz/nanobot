"""Utility for importing classes or objects from a dotted path."""

import importlib
from typing import Any

def import_from_dotted_path(dotted_path: str) -> Any:
    """Import a class or object from a dotted string path.

    Args:
        dotted_path: e.g. "mypackage.sql_memory:SQLMemoryPlugin" or "mypackage.module.ClassName"

    Returns:
        The imported object.
    """
    if ":" in dotted_path:
        module_path, obj_name = dotted_path.split(":", 1)
    else:
        module_path, obj_name = dotted_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
