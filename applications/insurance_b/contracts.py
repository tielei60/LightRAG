"""Select an explicitly frozen contract without importing the live A application."""

import importlib


def contract_module(settings, name):
    package = (
        "_current" if settings.get("CONTRACT_REVISION") == "current" else "_frozen"
    )
    return importlib.import_module(f"applications.insurance_b.{package}.{name}")
