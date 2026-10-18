# MLP-Mixer Compaction Experiments

Reduce an MLP-Mixer for CIFAR-10 by removing hidden neurons and complete Mixer blocks. Compare accuracy, parameter count, and GPU inference latency for the original, zeroed dense, and physically compact models, using PyTorch eager execution and `torch.compile`.

## Setup

Run from the repository root in a **WSL2 Ubuntu Bash terminal**. Training runs remotely on a Google Colab GPU; no local GPU is required.

Install the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) and configure authentication following its instructions:

```bash
uv tool install google-colab-cli
```

The remote runner uses Colab's installed PyTorch and CUDA environment. Local installation of `requirements.txt` is not needed.

## Run

```bash
# Quick GPU check with synthetic data; no dataset download
bash start_colab_wsl.sh smoke

# Full CIFAR-10 experiment
bash start_colab_wsl.sh full
```

The full run downloads CIFAR-10 from a checksum-verified mirror, trains three seeds (`123,456,789`), and evaluates 60–85% hidden-width reduction targets plus removal of one or two Mixer blocks. It benchmarks batch sizes 1, 128, and 1024. Model selection uses validation data; the selected model is then evaluated on the held-out test set.

Default GPU selector: **G4**. To customize a run:

```bash
MIXER_COLAB_GPU=G4 \
MIXER_SEEDS=123,456 \
MIXER_LEVELS=0.60,0.75,0.85 \
MIXER_BASELINE_EPOCHS=40 \
MIXER_FINETUNE_EPOCHS=15 \
bash start_colab_wsl.sh full
```

Use `MIXER_DEPTH_REMOVALS=` to disable block-removal experiments. Run `bash start_colab_wsl.sh --help` for supported settings.

## Results and shutdown

Results are downloaded to `artifacts/lottery-mixer-..._results.zip`. Full runs include:

- `repeat_summary.csv`: accuracy, model size, and speedups.
- `repeat_results.json`: detailed measurements and held-out test results.
- Per-seed checkpoints, logs, configuration, and environment metadata.

The wrapper stops its remote session after completion or interruption with **Ctrl+C**. If automatic shutdown fails, use the session name printed in the terminal:

```bash
colab stop -s <session-name>
```

Smaller models do not necessarily run proportionally faster on a GPU. Accuracy comparisons include additional fine-tuning; only the zeroed and compact controls represent the same final subnetwork.
