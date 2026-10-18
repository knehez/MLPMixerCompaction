"""Exercise remote-session lifecycle without allocating a Colab GPU."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import contextlib
import io
import zipfile

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.name == "nt", "Run shell workflow tests in WSL")
class ShellWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mixer workflow ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/run_mixer_colab.sh", self.root / "scripts")
        (self.root / "mlp_mixer_cifar10.py").touch()
        (self.root / "scripts/colab_run_mixer.py").touch()
        cli = self.root / "colab"
        cli.write_text('''#!/usr/bin/env bash
echo "$*" >> "$MOCK_LOG"
case "$1" in
  upload) exit "${MOCK_UPLOAD_STATUS:-0}" ;;
  exec) exit "${MOCK_EXEC_STATUS:-0}" ;;
  download) exit "${MOCK_DOWNLOAD_STATUS:-0}" ;;
esac
''', encoding="utf-8")
        cli.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                        MOCK_LOG=str(self.root / "calls.log"))

    def run_script(self, *args, **env):
        result = subprocess.run(["bash", str(self.root / "scripts/run_mixer_colab.sh"), *args],
                                env=dict(self.env, **env), capture_output=True, text=True)
        log = self.root / "calls.log"
        calls = log.read_text().splitlines() if log.exists() else []
        return result, calls

    def test_success_downloads_before_stop(self):
        result, calls = self.run_script("full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line.split()[0] for line in calls],
                         ["new", "upload", "exec", "download", "stop"])
        self.assertIn("MIXER_MODE=full", calls[2])

    def test_execution_failure_downloads_diagnostics_and_stops(self):
        result, calls = self.run_script(MOCK_EXEC_STATUS="7")
        self.assertEqual(result.returncode, 7)
        self.assertTrue(calls[-2].startswith("download "))
        self.assertTrue(calls[-1].startswith("stop "))

    def test_upload_failure_stops_without_execution(self):
        result, calls = self.run_script(MOCK_UPLOAD_STATUS="8")
        self.assertEqual(result.returncode, 8)
        self.assertEqual([line.split()[0] for line in calls], ["new", "upload", "stop"])

    def test_download_failure_is_reported(self):
        result, calls = self.run_script(MOCK_DOWNLOAD_STATUS="9")
        self.assertEqual(result.returncode, 9)
        self.assertTrue(calls[-1].startswith("stop "))

    def test_no_download(self):
        result, calls = self.run_script("smoke", "no-download")
        self.assertEqual(result.returncode, 0)
        self.assertFalse(any(line.startswith("download ") for line in calls))
        self.assertIn("MIXER_ARCHIVE_RESULTS=0", calls[2])

    def test_mirror_is_forwarded(self):
        result, calls = self.run_script("full", MIXER_CIFAR10_URL="https://mirror.example/cifar.tar.gz")
        self.assertEqual(result.returncode, 0)
        self.assertIn("MIXER_CIFAR10_URL=https://mirror.example/cifar.tar.gz", calls[2])

    def test_repeat_settings_are_forwarded(self):
        result, calls = self.run_script('full', MIXER_SEEDS='123,456', MIXER_LEVELS='0.60,0.85')
        self.assertEqual(result.returncode, 0)
        self.assertIn('MIXER_SEEDS=123,456', calls[2])
        self.assertIn('MIXER_LEVELS=0.60,0.85', calls[2])

    def test_depth_can_be_disabled_explicitly(self):
        result, calls = self.run_script('full', MIXER_DEPTH_REMOVALS='')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--env MIXER_DEPTH_REMOVALS= ', calls[2])

    def test_invalid_arguments_never_allocate(self):
        result, calls = self.run_script("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


class RemoteDiagnosticsTests(unittest.TestCase):
    def test_missing_gpu_preserves_log_manifest_and_source(self):
        spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/colab_run_mixer.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mlp_mixer_cifar10.py"
            source.write_text("# exact uploaded source\n", encoding="utf-8")
            def remote_path(value):
                return root / str(value).removeprefix("/content/")
            fake_torch = types.SimpleNamespace(__version__="test", version=types.SimpleNamespace(cuda=None),
                                              cuda=types.SimpleNamespace(is_available=lambda: False))
            real_zip = zipfile.ZipFile
            def remote_zip(filename, *args):
                return real_zip(remote_path(filename), *args)
            with patch.object(runner, "Path", remote_path), \
                 patch.object(runner.zipfile, "ZipFile", remote_zip), \
                 patch.dict(sys.modules, torch=fake_torch,
                            torchvision=types.SimpleNamespace(__version__="test")), \
                 patch.dict(os.environ, MIXER_MODE="smoke", MIXER_ARCHIVE_RESULTS="1"), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    runner.main()
            with real_zip(root / "mixer_results.zip") as archive:
                manifest = json.loads(archive.read("run_manifest.json"))
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("CUDA unavailable", archive.read("run.log").decode())
                self.assertEqual(archive.read(source.name), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
