"""Exercise the single insurance package in fresh interpreters, without services."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from applications.insurance.paths import INSURANCE_DIR, PROMPT_PROFILE_DIR, REPO_ROOT

APPLICATION_MODULES = (
    "answer_audit",
    "answer_query",
    "database_storage",
    "evidence_extract",
    "evidence_import",
    "evidence_semantics",
    "extraction_profile",
    "formula_consistency",
    "grounded_query",
    "manage",
    "prepare_batch",
    "prepare_context",
    "query_trace",
    "server",
    "source_scope",
    "verify_batch",
    "workspace_registry",
)


def run_python(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def legacy_imports(source: str) -> list[str]:
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        else:
            continue
        imports.extend(
            name for name in names if name == "product" or name.startswith("product.")
        )
    return imports


@pytest.mark.parametrize("module", APPLICATION_MODULES)
def test_application_module_imports_without_legacy_package(module: str):
    result = run_python(
        "-c",
        f"""
import importlib
import sys
from pathlib import Path
from applications.insurance.paths import INSURANCE_DIR
name = 'applications.insurance.{module}'
module = importlib.import_module(name)
assert module.__name__ == name
assert module.__spec__.name == name
assert Path(module.__file__).resolve() == INSURANCE_DIR / '{module}.py'
assert not any(name == 'product' or name.startswith('product.') for name in sys.modules)
if {module!r} == 'evidence_extract':
    profile = importlib.import_module('applications.insurance.extraction_profile')
    assert module.ExtractionProfile is profile.ExtractionProfile
if {module!r} == 'evidence_import':
    assert module.manage is importlib.import_module('applications.insurance.manage')
if {module!r} == 'server':
    answer = importlib.import_module('applications.insurance.answer_query')
    assert module.AnswerQuery is answer.AnswerQuery
""",
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr


@pytest.mark.parametrize(
    "module",
    ("manage", "prepare_batch", "evidence_import", "query_trace", "verify_batch"),
)
def test_application_command_help_without_execution(module: str):
    result = run_python("-m", f"applications.insurance.{module}", "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "RuntimeWarning" not in result.stderr
    assert "Traceback" not in result.stderr


def test_manage_direct_script_help():
    result = run_python(str(INSURANCE_DIR / "manage.py"), "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_manage_keeps_redacted_failure_output():
    result = run_python(
        "-m",
        "applications.insurance.manage",
        "--batch",
        "missing-migration-fixture",
        "preview",
    )
    assert result.returncode == 1
    assert "Operation stopped (FileNotFoundError)" in result.stderr
    assert "Traceback" not in result.stderr
    assert str(REPO_ROOT) not in result.stderr


def test_makefile_uses_application_command_and_master_default():
    result = subprocess.run(
        ["make", "-n", "-f", "applications/insurance/Makefile", "status"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "-m applications.insurance.manage" in result.stdout
    assert '--batch "master" status' in result.stdout
    assert str(REPO_ROOT / ".venv/bin/python") in result.stdout


def test_repository_and_resource_paths_follow_application_package():
    assert INSURANCE_DIR == REPO_ROOT / "applications/insurance"
    assert Path(__file__).resolve().parent == INSURANCE_DIR
    assert (REPO_ROOT / "lightrag").is_dir()
    assert (PROMPT_PROFILE_DIR / "entity_type/products.yml").is_file()
    assert (PROMPT_PROFILE_DIR / "user_prompt/products.md").is_file()


def test_legacy_package_is_absent():
    assert not (REPO_ROOT / "product").exists()
    result = run_python(
        "-c",
        "import importlib.util; assert importlib.util.find_spec('product') is None",
    )
    assert result.returncode == 0, result.stderr


def test_production_modules_do_not_import_legacy_package():
    offenders = {
        source.name: imports
        for source in sorted(INSURANCE_DIR.glob("*.py"))
        if not source.name.startswith("test_")
        and (imports := legacy_imports(source.read_text()))
    }
    assert not offenders


@pytest.mark.parametrize(
    "source",
    (
        "import product",
        "import product.evidence_extract as extraction",
        "from product import manage",
        "from product.answer_query import AnswerQuery",
    ),
)
def test_legacy_dependency_guard_detects_each_import_form(source: str):
    assert legacy_imports(source)


def test_legacy_dependency_guard_allows_application_imports_and_prose():
    source = """
from applications.insurance import manage
from applications.insurance.evidence_extract import EvidenceUnit
product = '示例产品'
# import product is a removed entry, not an executable dependency.
"""
    assert legacy_imports(source) == []
