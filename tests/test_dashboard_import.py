import subprocess
import sys
from pathlib import Path


def test_dashboard_imports_when_working_directory_is_app_folder():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import dashboard"],
        cwd=project_root / "app",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
