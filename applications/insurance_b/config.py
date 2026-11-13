"""Explicit B configuration; no discovery of A files or database targets."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .common import inside

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    values: dict[str, str]
    root: Path = ROOT

    def __post_init__(self):
        if self.get("CONTRACT_REVISION") not in {"", "current"}:
            raise ValueError("Unknown frozen contract revision")
        if self.get("PREPARATION", "native") not in {"native", "business"}:
            raise ValueError("Unknown preparation arm")
        if not re.fullmatch(r"insurance_b[a-z0-9_-]*", self.workspace):
            raise ValueError("B_WORKSPACE must start with insurance_b")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.run_id):
            raise ValueError("B_RUN_ID is invalid")
        if (
            self.get("POSTGRES_HOST") != "127.0.0.1"
            or self.integer("POSTGRES_PORT") != 15433
        ):
            raise ValueError(
                "B only connects to its independent PostgreSQL on 127.0.0.1:15433"
            )
        if self.get("POSTGRES_DATABASE") != "insurance_b":
            raise ValueError("B PostgreSQL database must be insurance_b")
        if self.get("NEO4J_URI") != "bolt://127.0.0.1:17687":
            raise ValueError("B only connects to its independent Neo4j on port 17687")
        if self.integer("HTTP_PORT") != 9625:
            raise ValueError("The B HTTP service uses its dedicated port 9625")
        if not 0 <= self.integer("P_OVERLAP") < self.integer("P_SIZE") <= 8192:
            raise ValueError("Invalid B native-P token parameters")
        for key in ["OPENAI_BASE_URL", "RERANK_URL", "DOCLING_ENDPOINT"]:
            parsed = urlsplit(self.get(key))
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
            ):
                raise ValueError(f"B_{key} must be an HTTP URL without credentials")
        for key in [
            "TOP_K",
            "CHUNK_TOP_K",
            "MAX_ENTITY_TOKENS",
            "MAX_RELATION_TOKENS",
            "MAX_TOTAL_TOKENS",
        ]:
            if self.integer(key) <= 0:
                raise ValueError(f"B_{key} must be positive")

    def get(self, key: str, default: str = "") -> str:
        return self.values.get("B_" + key, default)

    def integer(self, key: str) -> int:
        return int(self.get(key))

    @property
    def workspace(self) -> str:
        return self.get("WORKSPACE")

    @property
    def run_id(self) -> str:
        return self.get("RUN_ID")

    @property
    def run_dir(self) -> Path:
        return inside(self.root, self.root / "output/insurance-b" / self.run_id)

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "input"

    @property
    def originals(self) -> Path:
        return inside(self.root, self.root / "inputs/insurance-b/originals")

    @property
    def profiles(self) -> Path:
        return self.root / (
            "profiles_current"
            if self.get("CONTRACT_REVISION") == "current"
            else "profiles"
        )

    @property
    def contract_snapshot(self) -> Path:
        name = "current" if self.get("CONTRACT_REVISION") == "current" else "baseline"
        return self.root / "snapshots" / (name + ".json")

    def model_key(self, role: str = "MODEL") -> str:
        return self.get(role.upper() + "_API_KEY") or self.get("MODEL_API_KEY")

    def require_models(self) -> None:
        if not self.model_key():
            raise ValueError(
                "Set B_MODEL_API_KEY in the independent config/b.env before model calls"
            )

    def public(self) -> dict:
        return {
            "B_DOCLING_MAX_POLLS": self.get("DOCLING_MAX_POLLS", "720"),
            **{
                key: value
                for key, value in self.values.items()
                if not any(x in key for x in ["API_KEY", "PASSWORD"])
            },
        }

    def activate(self) -> None:
        """Set native SDK variables only inside this B process."""
        values = {
            "PYTHON_DOTENV_DISABLED": "1",
            "WORKSPACE": self.workspace,
            "WORKING_DIR": str(self.root / "rag_storage"),
            "INPUT_DIR": str(self.input_dir),
            "PROMPT_DIR": str(self.profiles),
            "ENTITY_TYPE_PROMPT_FILE": "products.yml",
            "USER_PROMPT_FILE": "products.md",
            "AUTH_ACCOUNTS": "",
            "LIGHTRAG_API_KEY": "",
            "SUMMARY_LANGUAGE": "Chinese",
            "TOP_K": self.get("TOP_K"),
            "CHUNK_TOP_K": self.get("CHUNK_TOP_K"),
            "MAX_ENTITY_TOKENS": self.get("MAX_ENTITY_TOKENS"),
            "MAX_RELATION_TOKENS": self.get("MAX_RELATION_TOKENS"),
            "MAX_TOTAL_TOKENS": self.get("MAX_TOTAL_TOKENS"),
            "VECTOR_CHUNK_CANDIDATE_MULTIPLIER": "4",
            "ENTITY_EXTRACTION_USE_JSON": "true",
            "LIGHTRAG_PARSER": "pdf:docling-P;md:native-P",
            "CHUNK_P_SIZE": str(self.integer("P_SIZE")),
            "CHUNK_P_OVERLAP_SIZE": str(self.integer("P_OVERLAP")),
            "CHUNK_P_DROP_REFERENCES": "false",
            "DOCLING_ENDPOINT": self.get("DOCLING_ENDPOINT"),
            "DOCLING_ENGINE_VERSION": self.get("DOCLING_ENGINE_VERSION"),
            "DOCLING_DO_OCR": "false",
            "DOCLING_FORCE_OCR": "false",
            "DOCLING_DO_FORMULA_ENRICHMENT": "false",
            "DOCLING_MAX_POLLS": self.get("DOCLING_MAX_POLLS", "720"),
            "EMBEDDING_USE_BASE64": self.get("EMBEDDING_USE_BASE64", "false"),
            "POSTGRES_HOST": self.get("POSTGRES_HOST"),
            "POSTGRES_PORT": self.get("POSTGRES_PORT"),
            "POSTGRES_DATABASE": self.get("POSTGRES_DATABASE"),
            "POSTGRES_USER": self.get("POSTGRES_USER"),
            "POSTGRES_PASSWORD": self.get("POSTGRES_PASSWORD"),
            "POSTGRES_WORKSPACE": self.workspace,
            "NEO4J_URI": self.get("NEO4J_URI"),
            "NEO4J_USERNAME": self.get("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": self.get("NEO4J_PASSWORD"),
            "NEO4J_DATABASE": "neo4j",
            "NEO4J_WORKSPACE": self.workspace,
        }
        os.environ.update(values)


def load(path: Path | None = None, *, root: Path = ROOT) -> Settings:
    selected = inside(root, path or root / "config/b.env")
    if not selected.is_file():
        raise ValueError("Create the independent B config/b.env first")
    values = {
        k: str(v)
        for k, v in dotenv_values(selected, interpolate=False).items()
        if v is not None
    }
    if any(not key.startswith("B_") for key in values):
        raise ValueError("The B configuration accepts only B_ variables")
    return Settings(values, root.resolve())
