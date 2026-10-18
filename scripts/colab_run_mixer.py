"""Run on a Google Colab VM via run_mixer_colab.sh."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile


SMOKE_CODE = '''
import sys
import copy
sys.path.insert(0, "/content")
import torch
from mlp_mixer_cifar10 import MLPMixer, compact_and_verify, benchmark_models
torch.manual_seed(123)
model = MLPMixer(depth=2, embedding_dim=32, token_dim=16, channel_dim=64).cuda()
x = torch.randn(4, 3, 32, 32, device="cuda")
optimizer = torch.optim.Adam(model.parameters())
loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3], device="cuda"))
loss.backward()
optimizer.step()
dense = copy.deepcopy(model).eval()
masks = []
with torch.no_grad():
    for block in model.mixer_blocks:
        mask = {}
        for name in ("token_mlp", "channel_mlp"):
            mlp = getattr(block, name)
            a, b = torch.ones_like(mlp.fc1.weight), torch.ones_like(mlp.fc2.weight)
            a[::2] = 0
            b[:, ::2] = 0
            mlp.fc1.weight *= a
            if mlp.fc1.bias is not None:
                mlp.fc1.bias[::2] = 0
            mlp.fc2.weight *= b
            mask[name + ".fc1"], mask[name + ".fc2"] = a, b
        masks.append(mask)
    model.eval()
    expected = model(x)
    pruned = compact_and_verify(model, masks, x)
    torch.testing.assert_close(pruned(x), expected, rtol=1e-4, atol=1e-5)
    assert sum(p.numel() for p in pruned.parameters()) < sum(p.numel() for p in model.parameters())
    torch.save(pruned.state_dict(), "smoke_pruned_model.pt")
benchmark_models(dense, pruned, masked=model, batch_sizes=(1, 128), warmup=5, iterations=5, rounds=3)
print("SMOKE OK: GPU training step, pruning equivalence and PyTorch speed benchmark", flush=True)
'''


def main():
    mode = os.environ.get("MIXER_MODE", "smoke")
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Invalid MIXER_MODE: {mode}")
    source = Path("/content/mlp_mixer_cifar10.py")
    output = Path("/content/mixer_results")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": mode, "status": "running", "python": sys.version,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    started = time.perf_counter()
    try:
        with (output / "run.log").open("w", encoding="utf-8") as log:
            def report(message):
                print(message, flush=True)
                print(message, file=log, flush=True)

            def run(command):
                with subprocess.Popen(
                    command, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                ) as process:
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                if code:
                    raise RuntimeError(f"Process failed with exit code {code}: {command}")

            try:
                import torch
                import torchvision

                manifest.update(torch=torch.__version__, torchvision=torchvision.__version__,
                                cuda=torch.version.cuda,
                                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; select a GPU Colab runtime.")
                manifest["gpu"] = torch.cuda.get_device_name(0)
                report(json.dumps(manifest))
                # Use Colab's PyTorch/torchvision; no ONNX dependencies needed.
                command = ([sys.executable, "-u", str(source)] if mode == "full"
                           else [sys.executable, "-u", "-c", SMOKE_CODE])
                manifest["command"] = command
                run(command)
                benchmark_path = output / 'benchmark_results.json'
                if not benchmark_path.is_file():
                    raise RuntimeError('Run completed without benchmark_results.json')
                manifest['benchmark'] = json.loads(benchmark_path.read_text(encoding='utf-8'))
                manifest["status"] = "success"
            except BaseException as exc:
                manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                report(traceback.format_exc())
                raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        if os.environ.get("MIXER_ARCHIVE_RESULTS", "1") == "1":
            with zipfile.ZipFile("/content/mixer_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for path in output.rglob('*'):
                    if path.is_file() and path.suffix in {".pt", ".log", ".json", ".csv"}:
                        archive.write(path, path.relative_to(output))
                if source.is_file():
                    archive.write(source, source.name)
            print("Results: /content/mixer_results.zip", flush=True)


if __name__ == "__main__":
    main()
