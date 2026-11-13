"""Offline fixtures; database and model calls are always replaced in unit tests."""

import shutil

import pytest
from dotenv import dotenv_values

from .config import ROOT, Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    values = {
        key: str(value or "")
        for key, value in dotenv_values(ROOT / "config/b.env.example").items()
    }
    values.update(
        B_MODEL_API_KEY="offline-test",
        B_POSTGRES_PASSWORD="offline-postgres",
        B_NEO4J_PASSWORD="offline-neo4j",
    )
    shutil.copytree(
        ROOT / "profiles",
        tmp_path / "profiles",
    )
    shutil.copytree(
        ROOT / "fixtures",
        tmp_path / "fixtures",
    )
    (tmp_path / "snapshots").mkdir()
    shutil.copyfile(
        ROOT / "snapshots/baseline.json", tmp_path / "snapshots/baseline.json"
    )
    monkeypatch.chdir(tmp_path)
    for key in [
        "PROMPT_DIR",
        "ENTITY_TYPE_PROMPT_FILE",
        "WORKSPACE",
        "POSTGRES_WORKSPACE",
        "NEO4J_WORKSPACE",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "INPUT_DIR",
        "WORKING_DIR",
        "SUMMARY_LANGUAGE",
        "USER_PROMPT_FILE",
        "ENTITY_EXTRACTION_USE_JSON",
        "LIGHTRAG_PARSER",
        "CHUNK_P_SIZE",
        "CHUNK_P_OVERLAP_SIZE",
        "CHUNK_P_DROP_REFERENCES",
        "DOCLING_ENDPOINT",
        "DOCLING_ENGINE_VERSION",
        "DOCLING_DO_OCR",
        "DOCLING_FORCE_OCR",
        "DOCLING_DO_FORMULA_ENRICHMENT",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield Settings(values, tmp_path)
