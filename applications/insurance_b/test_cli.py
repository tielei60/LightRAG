from .cli import parser, resolve_cli_paths
from .config import PROJECT_ROOT, ROOT


def test_repository_relative_result_paths_survive_b_cwd_change(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    args = resolve_cli_paths(
        parser().parse_args(
            [
                "compare",
                "--a",
                "applications/insurance_b/output/a.json",
                "--b",
                "applications/insurance_b/output/b.json",
            ]
        )
    )
    assert args.a == ROOT / "output/a.json"
    assert args.b == ROOT / "output/b.json"
