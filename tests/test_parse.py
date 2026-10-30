import unittest

from cortex.evals.parse import parse_files


class TestParse(unittest.TestCase):
    def test_single_file(self):
        out = "Here you go:\n\n### calc.py\n```python\ndef f():\n    return 1\n```\n"
        r = parse_files(out, ["calc.py"])
        self.assertEqual(r.files, {"calc.py": "def f():\n    return 1\n"})
        self.assertFalse(r.format_error)

    def test_multiple_files_and_backticked_header(self):
        out = "### `a.py`\n```\nA = 1\n```\n\n### b/c.py\n```py\nB = 2\n```"
        r = parse_files(out, ["a.py", "b/c.py"])
        self.assertEqual(r.files, {"a.py": "A = 1\n", "b/c.py": "B = 2\n"})

    def test_inner_fence_is_kept(self):
        body = 'DOC = """\n```\nexample\n```\n"""\nX = 1'
        out = f"### doc.py\n```python\n{body}\n```\n"
        r = parse_files(out, ["doc.py"])
        self.assertEqual(r.files["doc.py"], body + "\n")

    def test_think_block_is_dropped(self):
        out = "<think>### calc.py\n```\nWRONG\n```\n</think>\n### calc.py\n```\nRIGHT = 1\n```"
        self.assertEqual(parse_files(out, ["calc.py"]).files["calc.py"], "RIGHT = 1\n")

    def test_disallowed_path_is_ignored(self):
        out = "### tests/test_x.py\n```\nx\n```\n### calc.py\n```\ny\n```"
        r = parse_files(out, ["calc.py"])
        self.assertEqual(list(r.files), ["calc.py"])
        self.assertEqual(r.ignored, ["tests/test_x.py"])

    def test_no_files_is_format_error(self):
        r = parse_files("I cannot do this.", ["calc.py"])
        self.assertTrue(r.format_error)

    def test_dot_slash_prefix(self):
        r = parse_files("### ./calc.py\n```\nz\n```", ["calc.py"])
        self.assertIn("calc.py", r.files)


if __name__ == "__main__":
    unittest.main()
