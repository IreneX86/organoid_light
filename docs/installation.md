# Installation and platform support

Use a dedicated **Python 3.11** environment. The frozen runtime uses TensorFlow
2.21.0, Keras 3.15.1, NumPy 2.4.6, tifffile 2026.3.3, and SciPy 1.16.2. Package requirements are
the authoritative pins. Do not silently downgrade TensorFlow to make an
unsupported platform install: that creates an unvalidated runtime.

## Supported platforms

| Platform | Route for this release | Qualification |
| --- | --- | --- |
| Linux x86-64 | CPU or NVIDIA GPU | Python 3.11; TensorFlow wheel requires glibc 2.27+ |
| Windows x86-64, native | CPU | TensorFlow 2.21 has a Windows CPU wheel; native CUDA support stopped after TensorFlow 2.10 |
| Windows with WSL2 Linux | CPU or NVIDIA GPU | Use the Linux environment and a working host NVIDIA/WSL driver |
| macOS 12+ Apple Silicon | CPU | Use native ARM64 Python; Metal is not in this release's validated runtime |
| macOS Intel | No native pinned-runtime route | TensorFlow 2.16 was the last official Intel macOS release; use a supported Linux machine instead |

These routes follow the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip)
and the [TensorFlow 2.21 wheel inventory](https://pypi.org/project/tensorflow/2.21.0/#files).
Wheel availability does not mean every platform has already completed this
project's end-to-end parity test; consult the [validation record](validation.md).

NumPy 2.4.6, SciPy 1.16.2, and tifffile 2026.3.3 require Python 3.11 or newer;
Python 3.11 is the common version selected for this distribution. Their primary
package metadata are available for
[NumPy](https://pypi.org/project/numpy/2.4.6/),
[SciPy](https://pypi.org/project/scipy/1.16.2/), and
[tifffile](https://pypi.org/project/tifffile/2026.3.3/).

## CPU installation

Clone the repository:

```bash
git clone https://github.com/IreneX86/organoid_light.git
cd organoid_light
```

On Linux or Apple Silicon macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[cpu]"
organoid-light doctor --device cpu
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install ".[cpu]"
.venv\Scripts\organoid-light.exe doctor --device cpu
.venv\Scripts\organoid-light.exe models download --organelle mito
.venv\Scripts\organoid-light.exe --organelle mito --input "D:\images\Mito" --output "D:\results\Mito_label" --device cpu
```

Using the environment executable directly avoids PowerShell activation-policy
changes. Native Windows may also need the Microsoft Visual C++ Redistributable
specified by TensorFlow's installation guide.

## NVIDIA GPU on Linux or WSL2

Create and activate the Python 3.11 environment above, then:

```bash
python -m pip install --upgrade pip
python -m pip install ".[gpu]"
nvidia-smi
organoid-light doctor --device gpu
```

The GPU extra uses TensorFlow's Linux pip CUDA dependency route. Before importing
TensorFlow in Linux GPU/auto mode, the CLI discovers this Python environment's
`nvidia/*/lib` directories, adds them to the process library search path, and
restarts itself once if needed. It preserves `CUDA_VISIBLE_DEVICES` and does not
edit system libraries or load CUDA modules. This reproduces the loader setup
verified in the research environment.

A working, compatible NVIDIA driver must already exist on the host. The “CUDA Version”
reported by `nvidia-smi` describes driver capability; it is not proof that the
TensorFlow process found its CUDA/cuDNN libraries. The doctor command checks
TensorFlow device initialization and a small GPU operation before a long batch.
The runtime uses one GPU within the existing visible-device restriction. It
does not combine all GPUs on the machine. Explicit GPU mode fails if no GPU is
available; automatic mode chooses CPU only when TensorFlow finds no GPU.

The new distribution's tests and their hardware scope are recorded in the
[validation report](validation.md). Reusing a previously working loader setup
does not establish that a fresh cloud host or every CUDA driver has passed a new
GPU test; run the doctor command on the actual deployment machine.

For WSL2, follow the
[TensorFlow Windows WSL2 instructions](https://www.tensorflow.org/install/pip#windows-wsl2)
and work inside the Linux environment. This package does not install drivers or
require an HPC scheduler, Runpod, a particular cloud template, or a system CUDA
module. On a shared GPU server, run within the allocation provided by your
scheduler and preserve its GPU visibility settings. Selecting `--device gpu`
does not grant access to an unallocated device.

If TensorFlow reports missing CUDA libraries, inspect the doctor's environment
report and the official guide before changing dependencies. Avoid mixing a random
system CUDA/cuDNN installation with the pinned pip environment. An unavailable
GPU can be bypassed explicitly with `--device cpu`; a GPU request should not be
mistaken for a successful CPU fallback.

## Download and run

```bash
organoid-light models download --organelle ld --models-dir models
organoid-light --organelle ld --input images/fluo --output masks/LD --models-dir models --device auto
```

The command downloads only the selected released weights and verifies SHA256.
Keep the downloaded files for subsequent offline runs. An HTTPS or checksum
failure should be resolved before inference; do not disable verification.

CPU inference is supported but can be slow, especially for the five-model LD
ensemble. Plan disk space for the model assets, TensorFlow and its optional GPU
libraries, input TIFFs, masks, and provenance records. Exact runtime and peak RAM
depend on image size, stack length, model choice, and hardware; this release does
not claim a universal minimum GPU-memory requirement or throughput.

If memory is limited, add `--batch-size 1` for a fresh run. The default is four
tiles, matching the frozen reference. Do not change the batch size when resuming
an existing output directory. Keep output and model-cache folders on a local
filesystem that supports hard links (for example, NTFS, APFS, or ext4); exFAT and
some network mounts cannot satisfy the atomic no-overwrite publication contract.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No matching distribution found` | Confirm Python 3.11, CPU architecture, platform, and recent pip. Intel macOS has no TensorFlow 2.21 wheel. |
| GPU list is empty | Run `doctor --device gpu`; check Linux/WSL2 driver and pip GPU dependencies. `nvidia-smi` alone is insufficient. |
| TIFF reports a missing compression codec | Export an uncompressed grayscale TIFF copy while preserving pixel values, or arrange a compatible TIFF codec dependency in a separately checked environment. |
| RGB or multi-channel input rejected | Export the intended single grayscale channel. Do not reinterpret a channel axis as Z. |
| Resume rejects a file | Inspect changed input/model/method/output hashes; use a new destination for a different run. |
| A large stack starts again after interruption | Resume is at TIFF-file boundaries. An unfinished multi-page stack is recomputed; complete files are verified and skipped. |
| Hard-link or filesystem operation fails | Use a local hard-link-capable output/cache folder; no partial final mask is adopted. |
| Output mask appears inverted | Inspect its values: 0 is background and 255 is foreground. Disable an inverted display LUT. |
| A biologically implausible result passes QC | QC checks integrity and geometry, not correct organelle identity. Compare the overlay with independent manual review. |

## Development tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The synthetic tests need neither a GPU nor downloaded checkpoints. Actual
checkpoint equivalence tests are recorded separately in
[validation.md](validation.md); do not report a fresh platform as validated solely
because packaging or synthetic tests pass.
