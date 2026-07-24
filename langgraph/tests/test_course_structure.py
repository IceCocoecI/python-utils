from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


COURSE_ROOT = Path(__file__).resolve().parents[1]


class CourseStructureTests(unittest.TestCase):
    def test_all_numbered_modules_have_docs_and_two_examples(self) -> None:
        modules = sorted(COURSE_ROOT.glob("[0-9][0-9]-*"))
        self.assertEqual(10, len(modules))
        for module in modules:
            with self.subTest(module=module.name):
                self.assertTrue((module / "README.md").is_file())
                examples = sorted((module / "examples").glob("*.py"))
                self.assertGreaterEqual(len(examples), 2)

    def test_top_level_does_not_shadow_installed_langgraph(self) -> None:
        self.assertFalse((COURSE_ROOT / "__init__.py").exists())
        spec = importlib.util.find_spec("langgraph.graph")
        self.assertIsNotNone(spec)
        self.assertIn("site-packages", str(spec.origin))

    def test_core_navigation_files_exist(self) -> None:
        names = {
            "README.md",
            "CONCEPT_MAP.md",
            "CHEATSHEET.md",
            "ENVIRONMENT.md",
            "VERSION_MATRIX.md",
            "TROUBLESHOOTING.md",
        }
        self.assertTrue(names.issubset({path.name for path in COURSE_ROOT.iterdir()}))

    def test_all_local_markdown_links_resolve(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(COURSE_ROOT / "scripts" / "check_links.py")],
            cwd=COURSE_ROOT.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            msg="\n".join((completed.stdout, completed.stderr)),
        )


if __name__ == "__main__":
    unittest.main()
