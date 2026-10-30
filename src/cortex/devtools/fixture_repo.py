"""Build a small synthetic git repo whose history exercises every miner path.

Expected mining result (oldest → newest commit):
  1. initial                      -> no_test_change (no parent: not even scanned)
  2. fix subtract sign bug        -> ACCEPT
  3. add multiply                 -> ACCEPT
  4. docs only                    -> no_test_change
  5. tests only                   -> tests_only
  6. add flaky counter feature    -> rejected: flaky_at_base
  7. big generated table          -> too_many_lines
  8. add clamp, but test unrelated-> rejected: base_not_failing
Test command for this repo: "{python} -m unittest -q {tests}"
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

TEST_COMMAND = "{python} -m unittest -q {tests}"

_GIT_ENV = {"GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}


def _commit(repo: Path, message: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    env = {**os.environ, **_GIT_ENV}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True, env=env)


CALC_V1 = '''def add(a, b):
    return a + b


def subtract(a, b):
    return b - a
'''
CALC_V2 = CALC_V1.replace("return b - a", "return a - b")
CALC_V3 = CALC_V2 + '''

def multiply(a, b):
    return a * b
'''
TEST_V1 = '''import unittest

from calc import add


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
'''
TEST_V2 = TEST_V1.replace("from calc import add", "from calc import add, subtract") + '''
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
'''
TEST_V3 = TEST_V2.replace("add, subtract", "add, multiply, subtract") + '''
    def test_multiply(self):
        self.assertEqual(multiply(4, 3), 12)
'''
TEST_V3B = TEST_V3 + '''
    def test_add_negative(self):
        self.assertEqual(add(-2, -3), -5)
'''
COUNTER = '''def bump():
    return 1
'''
# Passes on odd-numbered runs only (state kept in the worktree) -> flaky everywhere.
FLAKY_TEST = '''import pathlib
import unittest


class TestFlaky(unittest.TestCase):
    def test_flaky(self):
        p = pathlib.Path(".flaky_counter")
        n = int(p.read_text()) + 1 if p.exists() else 1
        p.write_text(str(n))
        self.assertEqual(n % 2, 1)
'''
CLAMP = CALC_V3 + '''

def clamp(x, lo, hi):
    return max(lo, min(x, hi))
'''
CLAMP_TEST = '''import unittest

from calc import add


class TestStillAdd(unittest.TestCase):
    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)
'''


def build(path: str | Path) -> Path:
    repo = Path(path)
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _commit(repo, "Initial calculator", {"calc.py": CALC_V1, "README.md": "# calc\n",
                                         "tests/test_calc.py": TEST_V1})
    _commit(repo, "Fix subtract sign bug\n\nsubtract(a, b) returned b - a.",
            {"calc.py": CALC_V2, "tests/test_calc.py": TEST_V2})
    _commit(repo, "Add multiply", {"calc.py": CALC_V3, "tests/test_calc.py": TEST_V3})
    _commit(repo, "Docs: usage", {"README.md": "# calc\n\nUsage: import calc\n"})
    _commit(repo, "More add tests", {"tests/test_calc.py": TEST_V3B})
    _commit(repo, "Add counter with flaky test", {"counter.py": COUNTER,
                                                  "tests/test_flaky.py": FLAKY_TEST})
    table = "TABLE = [\n" + "".join(f"    {i},\n" for i in range(400)) + "]\n"
    _commit(repo, "Big generated table", {"table.py": table,
                                          "tests/test_table.py": "import unittest\n"})
    _commit(repo, "Add clamp (test does not cover it)", {"calc.py": CLAMP,
                                                         "tests/test_still_add.py": CLAMP_TEST})
    return repo


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "fixture-repo"))
