# Frozen model cards — `models-v1`

This release preserves three operational inference methods from the Organoid
research project. “Frozen” means that checkpoint identity and processing rules
are fixed. It does not imply that every method has independent external
validation. The [package model manifest](../src/organoid_light/model_manifest.json)
is authoritative for released asset names, SHA256, and inference settings.
The checkpoints are exported for inference with optimizer state removed and
learned weights preserved. Original research-checkpoint hashes below identify
source provenance; released-file hashes consequently differ and are checked
against the manifest.

## Shared scope

The task is binary semantic segmentation of single-channel microscopy images.
Each plane is normalized independently between its 1st and 99.8th intensity
percentiles, clipped to the model range, and predicted using 512×512 overlapping
tiles with 128-pixel overlap. Blended probabilities are returned to the original
pixel grid without whole-image resizing. The reference batch size is four tiles;
the CLI permits smaller batches for memory-constrained inference. Batch size is
recorded in provenance because backend roundoff can differ.

Multi-page stacks and sequential TIFF folders receive independent 2D predictions.
The methods use no neighboring slice, organoid identity, temporal information,
or 3D reconstruction. Outputs are `uint8` TIFFs, background 0 and foreground 255.
They are predictions for human review, not manual labels, instance IDs, or
validated object counts.

The research datasets are not distributed. Source acquisition, staining,
microscope, resolution, and annotation convention may differ from a new user's
images. Before quantitative use, assess complete independently annotated images
from the intended deployment conditions. Preserve source-separated evaluation
and do not select thresholds on a final test set.

## Mito: baseline fold-0 2D U-Net

| Field | Value |
| --- | --- |
| Target | Mitochondrial foreground |
| Checkpoint | Frozen baseline fold 0; one 2D U-Net |
| Original checkpoint SHA256 | `66db8edc3006a9b70fdd75901f6f5e4a40359699f7a372d28f054d8dec58ecd3` |
| Decision | Foreground probability ≥ 0.45 |
| Extra morphology | None |
| Training context | 36 manually labelled development images, internally split |

This is the checkpoint used for the existing operational Mito predictions. It is
**not** a final model retrained on all 36 images and is **not** a five-model
ensemble. A single fold-0 internal evaluation used 22 training, 7 validation,
and 7 held-out images, with validation-selected threshold 0.45.

For context, the original unchanged five-fold baseline across all 36 development
images reported mean per-image OOF Dice **0.8543 ± 0.0628** and mean IoU **0.7507**.
The SD is image-to-image sample SD, not a confidence interval. Those numbers
describe five different fold models, not external performance of this one
distributed checkpoint. Fold 0's seven held-out images had mean Dice **0.8693**.

Original organoid/Z identities are unavailable for the development pool, so
related-image leakage across image-level folds cannot be excluded. Repeated
research inspection further limits an unbiased generalization interpretation.
Sparse foreground images showed lower precision and Dice. Inspect small,
elongated structures and background false positives on new acquisitions.

## LD: focal five-model ensemble, 1/5 vote

| Field | Value |
| --- | --- |
| Target | Lipid-droplet foreground |
| Checkpoints | Five separately trained focal-loss 2D U-Nets, folds 0–4 |
| Fold probability thresholds | `[0.65, 0.80, 0.60, 0.35, 0.50]` |
| Decision | Foreground if **at least one** fold votes foreground |
| Extra morphology | None |
| Training context | 36 manually labelled development images; derived binary labels retained in the research project |

Each model first thresholds its full-size probability map using its own locked
validation threshold. The final binary result is the union of those five masks.
This is not averaging probabilities and applying threshold 0.2, and it is not
the earlier 3/5 voting preview.

The operator preferred 1/5 voting after inspecting real target predictions,
especially small droplets. That supports use as a review aid but does not
establish improved precision, Dice, or performance on independently acquired
images. In the original focal experiment, all five preregistered inner-validation
gates rejected focal; raw candidate OOF Dice was approximately **0.7627**, versus
**0.7637** for the frozen LD baseline. These repeatedly inspected internal scores
do not evaluate this later 1/5 union policy.

The reviewed inference method is distributed deliberately and labelled as such;
it does not retroactively overturn the research selection decisions. Union
voting can retain additional weak candidates and false positives. Tiny dots,
bright debris, and changes in pixel size need manual review. Binary connected
regions should not automatically be interpreted as individual droplets.

## Nuc: internal U-Net plus reviewed morphology

| Field | Value |
| --- | --- |
| Target | Nuclear foreground, following the original annotation convention |
| Checkpoint | Original internal Nuc 2D U-Net |
| Original checkpoint SHA256 | `c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e` |
| Candidate/core probability thresholds | 0.03 / 0.45 |
| Minimum candidate size | 500 pixels, 8-connected components |
| Closing and smoothing | Radius-4 bounded closing; 5×5 median with topology safeguards |
| Separation | Conservative curved split guided by exactly two substantial predicted cores |
| Training context | 167 labelled images; extensive near-duplicates and incomplete source provenance |

**Nuc lacks reliable cross-organoid external validation.** Same-batch internal
validation does not establish accuracy on unrelated organoids. The 167 images
cannot currently be reliably mapped to original organoid/Z sources. Some labels
were decided with visual context from nearby slices, while this model observes
only one plane. These limitations can affect both evaluation independence and
whether the single-image target is identifiable.

The reviewed transformation removes components below 500 pixels, fills enclosed
holes, and applies closing/smoothing while protecting separate components. When
a cleaned candidate contains exactly two substantial model-predicted strong
cores, it may be divided along a curved gap. A split is retained only if the
required geometric conditions pass; ambiguous cases remain unsplit. This is a
review postprocessor, not a trained instance-segmentation model.

The low candidate threshold and morphology were chosen after inspecting target
images. Those viewed images are therefore unsuitable as an untouched external
test set for this choice. Smooth, filled shapes can look plausible while still
being biologically wrong. Small nuclei may be removed; touching nuclei may
remain merged; faint nuclei may still be missed; bright non-nuclear structures
may survive. Pixel-based rules do not transfer automatically across magnification
or resolution. The binary output does not assign a distinct integer ID to each
nucleus.

## Provenance, licensing, and interpretation

Inference loads only the released, hash-verified model assets. Do not substitute
another checkpoint under the same filename. Asset identity and algorithm parity
are engineering properties; they are separate from biological accuracy.

Refer to the repository and model-release notices for applicable licensing terms.
The fact that an asset is publicly downloadable does not establish permission to
redistribute its source training data, which are not included here. The package
does not infer or assert consent, image ownership, or clinical validity from a
TIFF file.
