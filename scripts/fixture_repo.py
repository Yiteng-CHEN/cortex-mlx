"""Create the synthetic repo used by tests and the pipeline demo: python scripts/fixture_repo.py <path>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortex.devtools.fixture_repo import TEST_COMMAND, build  # noqa: E402

if __name__ == "__main__":
    repo = build(sys.argv[1] if len(sys.argv) > 1 else "fixture-repo")
    print(f"created {repo}\n\nadd to config/local.toml:\n")
    print(f'[repos.fixture]\npath = "{repo.resolve()}"\ntest = "{TEST_COMMAND}"')
