# organoid_light

Portable inference for three frozen organelle segmentation methods: mitochondria
(`mito`), lipid droplets (`ld`), and nuclei (`nuc`). Give it a grayscale TIFF or a
folder of TIFFs and receive binary masks with the **same filenames, dimensions,
slice order, and pixel coordinates**. Background is **0 / black**; foreground is
**255 / white**.

This is an inference distribution, with model weights delivered through
[GitHub Releases](https://github.com/IreneX86/organoid_light/releases/tag/models-v1).
It contains no training images, manual masks, experiment runs, or cloud-service
credentials. You can run it locally without a Runpod account.

**The models are research tools, and their outputs require human review. Nuc
currently has no reliable external validation across independent organoids.**
The LD method is the operator-preferred five-model union, not a demonstrated
improvement over the research baseline. See the [model cards](docs/model-cards.md)
before interpreting measurements from any predicted mask.

## Quick start

Use **Python 3.11**. After cloning this repository, create and activate a virtual
environment as described in [installation](docs/installation.md), then run:

```bash
python -m pip install ".[cpu]"
organoid-light models download --organelle mito
organoid-light --organelle mito --input data/Mito --output results/Mito_label --device cpu
```

For an NVIDIA GPU on supported Linux or Windows WSL2, install the `gpu` extra
instead of `cpu` and verify the environment:

```bash
python -m pip install ".[gpu]"
organoid-light doctor --device gpu
organoid-light --organelle mito --input data/Mito --output results/Mito_label --device gpu
```

`--device gpu` requires usable GPU execution; `--device cpu` runs on CPU.
`--device auto` uses the first GPU visible to TensorFlow when one is available,
otherwise CPU. Only one visible GPU is used; existing `CUDA_VISIBLE_DEVICES`
restrictions are respected. Installation and driver
requirements differ by operating system; native Windows and Apple Silicon macOS
use the CPU route in this release.

## Choose an organelle

```bash
organoid-light models download --organelle all

organoid-light --organelle mito --input data/Mito --output results/Mito_label --device auto
organoid-light --organelle ld --input data/fluo --output results/fluo_label_vote1 --device auto
organoid-light --organelle nuc --input data/fluo --output results/Nuc_label --device auto
```

| Choice | Frozen method in `models-v1` | What to review |
| --- | --- | --- |
| `mito` | One 2D U-Net, baseline fold 0, probability threshold 0.45 | Thin branches, sparse-image false positives, imaging-domain changes |
| `ld` | Five focal U-Nets, validation-locked fold thresholds, at least 1/5 votes | Small droplets and false positives; union voting prioritizes sensitivity |
| `nuc` | Original 2D Nuc U-Net plus reviewed low/core thresholds and conservative morphology | Missed nuclei, bright debris, boundary changes, unresolved touching nuclei |

All three use per-plane p1–p99.8 normalization, overlapping 512×512 tiles,
128-pixel overlap, and blended full-size probabilities. The reference tile batch
size is four; `--batch-size 1` can reduce inference memory. Images are **not resized**
to a smaller whole-image grid. Method settings are fixed in this release;
there is no threshold-tuning command. LD and Nuc do not reuse Mito weights.

## Input and output

For a folder:

```text
data/Mito/                    results/Mito_label/
  Mito_001.tif       ->          Mito_001.tif
  Mito_002.tif       ->          Mito_002.tif
```

The output also includes `organoid-light.json` (provenance and resume state) and
`quality.csv` (quality-control records). Keep these beside the masks: they
identify the input hashes, released weights, processing method, and output
hashes used by safe resume.

Supported `.tif` / `.tiff` inputs are single-channel 2D TIFFs and unambiguous single-channel
multi-page TIFF stacks. A folder containing a continuous Z-stack as separate
TIFFs is processed file by file. A multi-page stack is processed page by page,
and saved with the same number and ordering of planes. RGB images, ambiguous
channel/time axes, and incompatible TIFF layouts are rejected instead of being
silently interpreted as Z slices. Export the desired grayscale channel first.
Hidden files and macOS AppleDouble sidecars such as `._image.tif` are excluded.
The supported axes are `YX`, `ZYX`, and ordinary grayscale page stacks `QYX` or
`IYX`, with one 2D plane per page. Spatial calibration and acquisition/OME
metadata are not copied into the mask: preserve them separately for measurements
in physical units.

```bash
organoid-light --organelle nuc --input data/nuc_stack.tif --output results/nuc_stack --device auto
organoid-light --organelle ld --input data/cells --output results/ld --recursive --device auto
```

With `--recursive`, relative subfolders are preserved. Choose an output directory
outside the input tree, and use a different output directory for each organelle
or model version. Paths containing spaces must be quoted.
The recursive input tree must contain only the grayscale channel appropriate to
the selected organelle. Do not point it at a mixed project tree containing Mito,
fluo, manual labels, or old prediction folders: filenames do not identify the
correct biological channel automatically.

**Stack support is slice-wise 2D inference.** The program does not infer missing
Z order, use neighboring planes, reconstruct a 3D volume, link objects across
planes, or provide instance IDs. A continuous acquisition therefore does not
make the model a 2.5D or 3D model.

## Resume an interrupted run

Repeat the same command and add `--resume`:

```bash
organoid-light --organelle nuc --input data/fluo --output results/Nuc_label --device cpu --resume
```

Resume verifies the recorded inputs, model/method identity, and completed output
hashes before skipping work. It does not assume that an arbitrary existing mask
is a valid prediction and does not silently overwrite changed files. A changed
input, mismatched method, or modified output requires investigation or a new
output directory. One process should own an output directory at a time.

A multi-page TIFF is one file transaction. If inference is interrupted midway
through that stack, resume recomputes the unfinished stack; it does not restart
at its last Z plane. Already verified complete TIFFs are skipped. Tile batch
size is part of run identity: changing `--batch-size` requires a fresh output
directory. Lower batch size can also change floating-point roundoff, so use the
default four when comparing against the frozen reference.

If the device or runtime changes between sessions, completed masks remain
verified and unchanged. The manifest retains `runtime_history`, and each newly
predicted file records its `runtime_id` into that history. This makes a
CPU/GPU-resumed batch distinguishable from one produced entirely on a single
backend; floating-point equivalence across those backends is not assumed.

## Model downloads and offline use

The small source repository does not contain model weights. Download commands
fetch the selected assets from the fixed `models-v1` Release and check their
SHA256 hashes against the installed
[model manifest](src/organoid_light/model_manifest.json) before use. A corrupt or
different checkpoint is rejected. Download just the organelle you need; LD
requires all five ensemble members.

All seven inference exports together are approximately **221 MB**; Mito and Nuc
each use one approximately 31.6 MB file, and LD uses five such files.

By default, models are cached in the operating system's user-cache directory
under `organoid-light/models-v1`. The release uses inference-only exports with
the learned weights preserved and training optimizer state removed.

For an explicit model location, use the same directory for both commands:

```bash
organoid-light models download --organelle nuc --models-dir models
organoid-light --organelle nuc --input data/fluo --output results/Nuc_label --models-dir models --device cpu
```

After downloading and verifying weights, inference can run offline. Do not rename
or replace downloaded assets to bypass the checksum check. A changed model needs
a new version and a new equivalence review.

Use a local filesystem that supports hard links for the output directory and
model cache, such as NTFS, APFS, or ext4. Publication relies on exclusive hard
links to prevent accidental overwrites. Some removable/network filesystems,
including exFAT, do not support this operation and will fail safely; choose a
compatible local folder instead.

## Quality checks and human review

The pipeline checks readable grayscale TIFFs, valid numeric inputs, preserved
geometry, binary output values, file integrity, and provenance. Quality-control
warnings identify inputs or predictions that deserve inspection. A technically
valid mask can still miss an organelle or mark the wrong structure: these checks
do **not** measure biological accuracy.

Open the original TIFF and its mask together in Fiji or another image viewer.
Inspect faint and bright regions, border objects, very sparse images, touching
objects, and images from each acquisition condition. Display adjustments are for
viewing; retain the original pixel values for inference. Do not turn automatic
masks into manual ground truth without independent annotation review.

Nuc's minimum-area and smoothing rules are expressed in **pixels**, not physical
units. Different magnification or pixel size can invalidate those rules. Nuc
splitting is a constrained review aid and is not a validated nucleus-counting
pipeline. LD's 1/5 vote may recover small candidates while also adding false
positives. None of the methods supplies a calibrated confidence or a guaranteed
external Dice score.

## Testing and reproducibility

```bash
python -m pip install ".[test]"
python -m pytest
```

The public tests use synthetic images and temporary files; they do not download
biological data or model weights. CI is configured for Python 3.11 on Linux,
Windows, and macOS. The [validation report](docs/validation.md) records the actual
frozen-reference comparisons performed for this distribution, with their scope
and limitations. Passing synthetic CI alone does not prove checkpoint parity or
accuracy on a new imaging domain.

Model hashes, preprocessing, tile order, probability blending, thresholds, and
postprocessing are fixed. Floating-point execution can vary across TensorFlow
builds and hardware; inspect near-threshold differences when moving between
backends. A method version should never be compared by filename alone.

## Frequently asked questions

**Can I use a Mito model for LD or Nuc?** No. The three selectors load separate
organelle-specific weights and output rules.

**Why are some small LDs or faint nuclei missed?** Training coverage, image
contrast, imaging-domain differences, and method choices all matter. Changing a
display brightness slider does not change the information supplied to the model.
Use independent annotations to assess a new dataset; this package does not tune
parameters on your images.

**Why is my mask black and white but not the same brightness as the input?** The
input may be 16-bit intensity data. The output is an 8-bit binary mask: 0 is
background, 255 is predicted organelle. Its shape and coordinates are preserved.

**Why did the GPU command stop?** Run `organoid-light doctor --device gpu` and
check [GPU installation](docs/installation.md#nvidia-gpu-on-linux-or-wsl2). A
visible GPU in `nvidia-smi` alone does not prove TensorFlow can initialize CUDA.

**Can I use Apple Metal or an Intel Mac?** Metal is not part of the tested frozen
runtime. TensorFlow 2.21 does not provide an Intel macOS wheel; see the platform
limits in [installation](docs/installation.md#supported-platforms).

**Are these production-validated models?** No. They reproduce the project's
approved operational methods. Mito is an internal fold-0 model; LD and Nuc
include qualitative review choices. See [model cards](docs/model-cards.md).

## Repository contents

`src/organoid_light/` contains the shared inference package and released model
manifest; `tests/` contains synthetic checks; `docs/` contains installation,
model limitations, and validation records. Model assets belong in Releases or a
local cache, never in Git. This repository does not train models, alter labels,
launch paid compute resources, or depend on the original research checkout.

## License

The package code and distributed inference weights are available under the
[MIT License](LICENSE). Training images and manual labels are not distributed
or licensed by this repository. Cite the model version and package version when
reporting results, and state any independent validation you performed.
