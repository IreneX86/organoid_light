# Release validation

The inference-only release was checked against the frozen research implementation
on **2026-10-19**. These checks establish implementation equivalence on the tested
cases. They do not establish segmentation accuracy on a new laboratory, organoid,
microscope or acquisition protocol.

## Checkpoint export

All seven source checkpoint SHA256 values matched the research records. Each
inference-only export retains the original architecture JSON (excluding the
removed compile configuration) and all **74 weight arrays / 7,851,969 weight
scalars** bit for bit. Optimizer state, training loss and compile configuration
were removed. No absolute private paths occur in the exported configuration.
Each asset is 31,571,794 bytes; the seven files total 221,002,558 bytes.

Each original/export pair also produced exactly equal probabilities for a fixed
synthetic 64×64 input on the same CPU runtime. Full details, source hashes and
release-asset hashes are in [model-export-verification.json](model-export-verification.json).

## Real-image comparison and command-line tests

The test set contains **three organelle cases using two unique real input TIFFs**:
one 1200×1200 Mito image, and one 1200×1200 fluorescence image used independently
for LD and Nuc. Inputs and historical masks remain private and are not bundled
with this repository. The public report records their hashes so the case
identities are auditable without distributing biological images.

For each case, the tool loaded the original research preprocessing and tiling
functions in isolation. It compared original and inference-only checkpoints on
the exact same normalized input, then compared the actual package `Segmenter`
with the reference pipeline. LD used all five validation-locked thresholds and
the reviewed 1/5 rule; Nuc used the original reviewed postprocessing function.

| Case | Original/export probability maximum error | Package/reference mask differing pixels | Historical/reference mask differing pixels |
| --- | ---: | ---: | ---: |
| Mito | 0.0 | 0 | 0 |
| LD, all five models | 0.0 for every model | 0 | 0 |
| Nuc | 0.0 | 0 | 0 |

All seven full-size probability maps were exactly equal, and all three final
masks were pixel-identical to both the same-runtime reference and the archived
delivered masks on these cases. Historical-output comparison is recorded
separately: differences from other hardware/runtime combinations must not be
silently presented as exact reproduction.

Each organelle also passed an actual fresh-process command-line run containing
a single TIFF and a two-plane Z-stack made by repeating its real input twice:

- The single output retained its filename and 1200×1200 dimensions.
- The stack retained its 2×1200×1200 dimensions, and both planes exactly matched
  the single-plane reference. This checks slice-wise execution and stack I/O;
  it is not a test of learned Z-context or biological continuity.
- All outputs passed `uint8`, 0/255, `MINISBLACK` and shape checks; quality reports
  and completed run journals were present.
- Rerunning the same command with `--resume` exited successfully, preserved mask
  SHA256 values and modification times, and did not initialize model inference.

The machine-readable record is [reference-validation.json](reference-validation.json).

## Runtime and coverage limits

The new full-image equivalence and CLI tests ran on **macOS arm64 CPU** with
Python 3.11.16, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, SciPy 1.16.2 and
tifffile 2026.3.3. This release test did not provision a new GPU. The checkpoint
export's smaller synthetic test used the existing TensorFlow 2.21.0 / Keras
3.15.1 CPU runtime with NumPy 2.5.3; full-image validation used the pinned release
environment above.

Exact equality here is evidence for these samples on this runtime. Floating-point
differences across TensorFlow kernels, CPU/GPU hardware or batch settings can
move pixels close to a threshold. The software test is intentionally separate
from the models' scientific limitations in [model-cards.md](model-cards.md).
In particular, Nuc has no reliable cross-organoid external validation, and the
LD 1/5 rule was chosen by visual review rather than independent ensemble testing.

The local lightweight suite passed **56 tests, with one case-sensitive filename
fixture skipped on the case-insensitive host filesystem, and 50 parameterized
subtests**. The explicit real-model checks described
above were run separately and passed. Ruff passed. Wheel/sdist builds and a
separate, non-editable wheel installation's model catalog, CPU doctor and a
63×71 synthetic TIFF prediction passed outside the source repository.
The first Windows CI run identified that syncing a read-only file descriptor is
unsupported on Windows. The TIFF temporary file now uses a write-capable handle
for the durability sync, with a regression test; mask calculations are unchanged.
Platform CI results are recorded separately below; the macOS checks alone do not
demonstrate every operating system/backend.

## Public model distribution

All seven assets were downloaded without GitHub authentication through the
installed CLI from the published `models-v1` Release. Every file passed its
registered size and SHA256 check. GitHub's server-side asset SHA256 digests also
matched the local export manifest. The private research checkout and original
checkpoint paths are not required to download these assets.

## Reproducing the private reference comparison

The repository includes `tools/validate_reference.py`. It accepts a private
source-map JSON list containing `id`, `path`, `sha256`, and `asset`, plus a private
cases JSON list containing `organelle`, `input`, and `golden_mask` (an optional
`id` gives the public case alias). Keep those path-bearing JSON files outside Git.

```bash
python tools/validate_reference.py \
  --research-code-dir /path/to/original/code \
  --source-map /private/source-map.json \
  --models-dir /path/to/downloaded/release-models \
  --cases /private/reference-cases.json \
  --output reference-validation.json
```

Use `--organelle mito`, `ld`, or `nuc` to run one case family at a time. The tool
does not alter source TIFFs, checkpoints or historical outputs, suppresses Python
bytecode writes in the reference tree, and removes its private temporary image
copies after success. On failure it retains private diagnostics locally and
writes a failed public report without embedding private paths or image pixels.
