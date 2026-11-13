from pathlib import Path

import pytest

from .config import Settings, load


@pytest.mark.parametrize(
    "key,value",
    [
        ("B_WORKSPACE", "master"),
        ("B_WORKSPACE", "../master"),
        ("B_POSTGRES_PORT", "5433"),
        ("B_POSTGRES_DATABASE", "lightrag_products"),
        ("B_POSTGRES_HOST", "remote"),
        ("B_NEO4J_URI", "bolt://127.0.0.1:7687"),
        ("B_HTTP_PORT", "9623"),
        ("B_RUN_ID", "../master"),
        ("B_P_OVERLAP", "2000"),
        ("B_P_SIZE", "9000"),
        ("B_OPENAI_BASE_URL", "https://name:secret@example.org/v1"),
        ("B_TOP_K", "0"),
    ],
)
def test_a_targets_and_invalid_configuration_are_rejected(settings, key, value):
    with pytest.raises(ValueError):
        Settings({**settings.values, key: value}, settings.root)


def test_config_path_cannot_escape_project(settings, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        load(Path("/tmp/not-the-b-project.env"), root=tmp_path)


def test_unprefixed_configuration_is_rejected(settings):
    target = settings.root / "config.env"
    target.write_text("WORKSPACE=master\n")
    with pytest.raises(ValueError, match="only B_"):
        load(target, root=settings.root)


def test_public_snapshot_redacts_all_credentials(settings):
    public = str(settings.public())
    assert "offline-test" not in public
    assert "offline-postgres" not in public
    assert "offline-neo4j" not in public


def test_models_require_b_credentials(settings):
    value = Settings({**settings.values, "B_MODEL_API_KEY": ""}, settings.root)
    with pytest.raises(ValueError, match="B_MODEL_API_KEY"):
        value.require_models()
