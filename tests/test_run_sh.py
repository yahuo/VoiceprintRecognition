"""启动脚本契约；仅在临时目录调用假的 Python，不加载模型或访问真实数据。"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class RunShTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "project with spaces"
        self.root.mkdir()
        self.cwd = Path(tmp.name) / "caller"
        self.cwd.mkdir()
        self.script = self.root / "run.sh"
        shutil.copy2(Path(__file__).resolve().parents[1] / "run.sh", self.script)
        self.python = self.root / "venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'pythonpath': os.environ.get('PYTHONPATH')}, ensure_ascii=False))\n"
            "raise SystemExit(int(os.environ.get('FAKE_PYTHON_EXIT', '0')))\n"
        )
        self.python.chmod(0o755)
        self.env = {**os.environ, "PYTHONPATH": "/test/extra", "FAKE_PYTHON_EXIT": "0"}

    def run_cli(self, *args):
        return subprocess.run(
            ["bash", str(self.script), *args], cwd=self.cwd, env=self.env,
            capture_output=True, text=True, timeout=10,
        )

    def test_routes_use_project_venv_and_preserve_relative_paths_and_opaque_id(self):
        cases = [
            (["register", "医生 A", "audio file.wav"],
             ["app.utils.voiceprint", "register", "--name", "医生 A", "--audio", "audio file.wav"]),
            (["identify", "audio file.wav"],
             ["app.utils.voiceprint", "identify", "--audio", "audio file.wav"]),
            (["list"], ["app.utils.voiceprint", "list"]),
            (["delete", " external/user/中文 id "],
             ["app.utils.voiceprint", "delete", "--id", " external/user/中文 id "]),
            (["meeting", "source/audio file.mp3"],
             ["app.services.meeting", "--audio", "source/audio file.mp3", "--output", "audio file_notes.md"]),
            (["live"], ["app.services.live", "--output", "live_meeting.md"]),
            (["live", "new live.md"], ["app.services.live", "--output", "new live.md"]),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(payload["args"], ["-m", *expected])
                self.assertEqual(payload["cwd"], str(self.cwd.resolve()))
                project_path, extra = payload["pythonpath"].split(os.pathsep, 1)
                self.assertEqual(Path(project_path).resolve(), self.root.resolve())
                self.assertEqual(extra, "/test/extra")

    def test_clean_is_rejected_without_deleting_audio_or_unowned_temp_files(self):
        for name in ("original.wav", "unowned.tmp", "review.md"):
            (self.cwd / name).write_bytes(b"keep original bytes")
        result = self.run_cli("clean")
        self.assertNotEqual(result.returncode, 0)
        for name in ("original.wav", "unowned.tmp", "review.md"):
            self.assertEqual((self.cwd / name).read_bytes(), b"keep original bytes")

    def test_missing_arguments_do_not_invoke_python(self):
        for args in ([], ["register"], ["register", "Name"], ["meeting"], ["identify"], ["delete"]):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"args":', result.stdout)
                self.assertNotIn("unbound variable", result.stderr)

    def test_help_does_not_require_venv(self):
        self.python.unlink()
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("delete [声纹ID]", result.stdout)
        self.assertNotIn("./run.sh clean", result.stdout)

    def test_missing_venv_does_not_fall_back_to_python_on_path(self):
        fallback = self.root / "bin"
        fallback.mkdir()
        shutil.move(self.python, fallback / "python")
        self.env["PATH"] = str(fallback) + os.pathsep + os.environ["PATH"]
        result = self.run_cli("list")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("缺少项目虚拟环境", result.stderr)
        self.assertNotIn('"args":', result.stdout)

    def test_python_failure_exit_status_is_preserved(self):
        self.env["FAKE_PYTHON_EXIT"] = "7"
        self.assertEqual(self.run_cli("list").returncode, 7)


if __name__ == "__main__":
    unittest.main()
