from types import SimpleNamespace

from .storage import _run, postgres_tcp_only


def test_migrated_long_paths_do_not_require_a_unix_socket(settings):
    path = settings.root / ".runtime/postgres/postgresql.conf"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# B database settings\nport=15433\nlisten_addresses='127.0.0.1'\nunix_socket_directories='/a/very/long/old/application/path'\n"
    )
    postgres_tcp_only(settings)
    configured = path.read_text()
    assert "unix_socket_directories=''" in configured
    assert "old/application/path" not in configured
    assert "port=15433" in configured
    assert "listen_addresses='127.0.0.1'" in configured
    postgres_tcp_only(settings)
    assert path.read_text() == configured


def test_database_launcher_detaches_from_the_callers_session(settings, monkeypatch):
    calls = []

    def launch(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            returncode=0, stdout="synthetic process started", stderr=""
        )

    monkeypatch.setattr("applications.insurance_b.storage.subprocess.run", launch)
    _run(settings, ["synthetic-database", "start"])
    assert calls[0]["start_new_session"] is True
