import subprocess
import sys

from .config import PROJECT_ROOT


def test_native_router_does_not_parse_or_replace_b_cli_arguments():
    script = """
import sys
from types import SimpleNamespace
from applications.insurance_b.server import native_routes
sys.argv = ['insurance-b', 'smoke', '--b-only-test-argument']
original = sys.argv
router = native_routes(SimpleNamespace(), 30)
assert sys.argv is original
assert sys.argv[1:] == ['smoke', '--b-only-test-argument']
assert any(route.path == '/query' for route in router.routes)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
