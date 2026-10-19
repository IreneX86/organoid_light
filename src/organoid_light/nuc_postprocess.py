"""Reviewed 2D nucleus-mask cleanup and conservative touching-nucleus separation.

The caller supplies two binary predictions from the *same* probability map:
the low-threshold foreground (0.03 in the reviewed preview) and the
high-threshold cores (0.45).  Cores are model predictions, not manual labels.
The function never reads source images, changes the trained model, or uses
adjacent slices.  Its output is a review mask, not validated ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


_CONNECT8 = np.ones((3, 3), dtype=bool)
_MIN_AREA = 500
_CLOSE_RADIUS = 4
_MEDIAN_SIZE = 5
_GAP_BASE = 1.2
_GAP_TAPER = 3.0
_GAP_TAPER_SCALE = 8.0
_SMOOTH_SIGMA = 2.0
_SMOOTH_THRESHOLD = 0.52
_SMOOTH_NEIGHBORHOOD = 12


def _binary(mask: np.ndarray, name: str) -> np.ndarray:
    """Accept bool or conventional binary TIFF values, never grayscale masks."""
    array = np.asarray(mask)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty 2D array")
    if array.dtype == np.bool_:
        return array.copy()
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have bool or integer binary values")
    values = set(np.unique(array).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"{name} must contain only 0/1 or 0/255")
    return array > 0


def _check_identity(before: np.ndarray, after: np.ndarray) -> int:
    """Require a one-to-one correspondence of retained 8-connected objects."""
    original, original_count = ndimage.label(before, structure=_CONNECT8)
    result, result_count = ndimage.label(after, structure=_CONNECT8)
    if original_count != result_count:
        raise ValueError(f"Component count changed: {original_count} -> {result_count}")
    if original_count == 0:
        return 0
    shared = (original > 0) & (result > 0)
    pairs = np.unique(np.column_stack((original[shared], result[shared])), axis=0)
    if (
        len(pairs) != original_count
        or len(np.unique(pairs[:, 0])) != original_count
        or len(np.unique(pairs[:, 1])) != result_count
    ):
        raise ValueError("A retained object was split, merged, or lost")
    if np.min(np.bincount(result.ravel())[1:]) < _MIN_AREA:
        raise ValueError("A retained object fell below 500 pixels")
    return result_count


def _guarded_close(base: np.ndarray, base_labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill holes and radius-4 clefts, reverting any closing-induced merge."""
    radius = _CLOSE_RADIUS
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    pad = radius + 2
    closing = ndimage.binary_closing(np.pad(base, pad), structure=disk)[pad:-pad, pad:-pad]
    filled = ndimage.binary_fill_holes(base)
    candidate = ndimage.binary_fill_holes(base | closing)
    candidate_labels, count = ndimage.label(candidate, structure=_CONNECT8)
    safe = candidate.copy()
    reverted_groups = 0
    for component_id in range(1, count + 1):
        region = candidate_labels == component_id
        origins = np.unique(base_labels[region])
        source_count = np.count_nonzero(origins)
        if source_count == 0:
            # Radius-4 closing can create a detached foreground island between
            # nearby objects. It has no source nucleus and must not become a
            # new component in the review mask.
            safe[region] = False
        elif source_count > 1:
            # The closing would join two original objects: retain only their
            # hole-filled shapes, rather than allowing a new bridge.
            safe[region] = filled[region]
            reverted_groups += 1
    if np.any(base & ~safe):
        raise ValueError("Closing unexpectedly removed original foreground")
    _check_identity(base, safe)
    return safe, reverted_groups


def _smooth_without_merging(closed: np.ndarray) -> np.ndarray:
    """Use the reviewed per-object remove-only median fallback."""
    labels, count = ndimage.label(closed, structure=_CONNECT8)
    result = np.zeros_like(closed)
    for component_id in range(1, count + 1):
        region = labels == component_id
        candidate = ndimage.median_filter(region, size=_MEDIAN_SIZE, mode="nearest") & region
        _, candidate_count = ndimage.label(candidate, structure=_CONNECT8)
        if candidate_count != 1 or np.count_nonzero(candidate) < _MIN_AREA:
            candidate = region
        result |= candidate
    _check_identity(closed, result)
    return result


def _meeting_boundary(labels: np.ndarray) -> np.ndarray:
    """Mark both sides of an 8-neighbor interface between the two territories."""
    boundary = np.zeros(labels.shape, dtype=bool)
    height, width = labels.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            first = (
                slice(max(0, dy), min(height, height + dy)),
                slice(max(0, dx), min(width, width + dx)),
            )
            second = (
                slice(max(0, -dy), min(height, height - dy)),
                slice(max(0, -dx), min(width, width - dx)),
            )
            boundary[first] |= (
                (labels[first] > 0)
                & (labels[second] > 0)
                & (labels[first] != labels[second])
            )
    return boundary


def _two_core_candidates(
    source_labels: np.ndarray, seed_labels: np.ndarray
) -> tuple[list[tuple[int, tuple[int, int]]], list[dict[str, object]]]:
    pairs: list[tuple[int, tuple[int, int]]] = []
    ambiguous: list[dict[str, object]] = []
    for component_id in range(1, int(source_labels.max()) + 1):
        region = source_labels == component_id
        strong = []
        for seed_id in np.unique(seed_labels[region]):
            if seed_id > 0 and np.count_nonzero(region & (seed_labels == seed_id)) >= _MIN_AREA:
                strong.append(int(seed_id))
        if len(strong) == 2:
            pairs.append((component_id, (strong[0], strong[1])))
        elif len(strong) > 2:
            ambiguous.append({
                "original_component_id": component_id,
                "seed_component_ids_at_threshold_0p45": strong,
                "reason": "more than two substantial cores; left unchanged",
            })
    return pairs, ambiguous


def _split_pair(
    original_component: np.ndarray,
    seed_labels: np.ndarray,
    pair_ids: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    ys, xs = np.nonzero(original_component)
    height, width = original_component.shape
    y0, y1 = max(0, int(ys.min()) - 12), min(height, int(ys.max()) + 13)
    x0, x1 = max(0, int(xs.min()) - 12), min(width, int(xs.max()) + 13)
    region = original_component[y0:y1, x0:x1]
    local_seed_labels = seed_labels[y0:y1, x0:x1]
    cores = [region & (local_seed_labels == seed_id) for seed_id in pair_ids]
    distance_a = ndimage.distance_transform_edt(~cores[0])
    distance_b = ndimage.distance_transform_edt(~cores[1])
    territory = np.where(region, np.where(distance_a <= distance_b, 1, 2), 0)
    line = _meeting_boundary(territory) & region
    if not line.any():
        raise ValueError("Two substantial cores have no meeting boundary")

    distance_to_line = ndimage.distance_transform_edt(~line)
    distance_to_background = ndimage.distance_transform_edt(region)
    gap_width = _GAP_BASE + _GAP_TAPER * np.exp(-distance_to_background / _GAP_TAPER_SCALE)
    gap = region & (distance_to_line <= gap_width)
    near_gap = ndimage.binary_dilation(gap, iterations=_SMOOTH_NEIGHBORHOOD)
    rebuilt = np.zeros(region.shape, dtype=bool)
    for side in (1, 2):
        child = region & (territory == side) & ~gap
        blurred = ndimage.gaussian_filter(
            child.astype(np.float32), sigma=_SMOOTH_SIGMA, mode="nearest"
        )
        smoothed = blurred >= _SMOOTH_THRESHOLD
        rebuilt |= np.where(near_gap, smoothed, child) & region & (territory == side) & ~gap

    # The archived method restores all 0.45 pixels, not only the two cores.
    # Candidate acceptance below fails closed if that reconnects the pair.
    rebuilt |= region & (local_seed_labels > 0)
    labels, count = ndimage.label(rebuilt, structure=_CONNECT8)
    removed_spurs = []
    for local_id in range(1, count + 1):
        piece = labels == local_id
        area = int(np.count_nonzero(piece))
        if area >= _MIN_AREA:
            continue
        if np.any(piece & (local_seed_labels > 0)):
            raise ValueError("A small detached piece contains a high-confidence core")
        removed_spurs.append(area)
        rebuilt[piece] = False

    labels, count = ndimage.label(rebuilt, structure=_CONNECT8)
    if count != 2:
        raise ValueError(f"Split produced {count} components instead of two")
    child_areas = np.bincount(labels.ravel())[1:]
    if np.min(child_areas) < _MIN_AREA:
        raise ValueError("A separated nucleus fell below the minimum area")
    core_destinations = []
    for core in cores:
        mapped = np.unique(labels[core])
        if len(mapped) != 1 or mapped[0] == 0:
            raise ValueError("A model core was erased or fragmented")
        core_destinations.append(int(mapped[0]))
    if len(set(core_destinations)) != 2:
        raise ValueError("The two model cores were assigned to one nucleus")
    if np.any(ndimage.binary_fill_holes(rebuilt) & ~rebuilt):
        raise ValueError("The separated pair acquired a hole")
    if np.any((local_seed_labels > 0) & region & ~rebuilt):
        raise ValueError("The split deleted a high-confidence pixel")

    return rebuilt, {
        "seed_component_ids_at_threshold_0p45": list(pair_ids),
        "gap_pixels": int(gap.sum()),
        "removed_unseeded_spur_areas": removed_spurs,
        "child_areas_pixels": child_areas.astype(int).tolist(),
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def postprocess_nucleus_masks(
    low_mask: np.ndarray, high_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a reviewed-style bool Nuc mask and JSON-safe processing report.

    The caller must threshold a single 2D model probability map at 0.03 and
    0.45.  Source components <500 px are removed before hole filling and
    guarded closing.  A two-core candidate is split only if every geometric
    and topology check succeeds; ambiguous/failed candidates stay unchanged.
    """
    low = _binary(low_mask, "low_mask")
    high = _binary(high_mask, "high_mask")
    if low.shape != high.shape:
        raise ValueError("low_mask and high_mask must have identical pixel grids")
    if np.any(high & ~low):
        raise ValueError("high_mask must be a subset of low_mask")

    raw_labels, raw_count = ndimage.label(low, structure=_CONNECT8)
    raw_areas = np.bincount(raw_labels.ravel())
    base = (raw_labels > 0) & (raw_areas[raw_labels] >= _MIN_AREA)
    base_labels, retained_count = ndimage.label(base, structure=_CONNECT8)
    closed, reverted_groups = _guarded_close(base, base_labels)
    cleaned = ndimage.median_filter(closed, size=_MEDIAN_SIZE, mode="nearest")
    smoothing_mode = "global_median5"
    try:
        before_count = _check_identity(base, cleaned)
        # The median can close a narrow background opening while preserving
        # the number of nuclei. Reject that candidate so review masks stay
        # hole-free without introducing foreground during smoothing.
        if np.any(ndimage.binary_fill_holes(cleaned) & ~cleaned):
            raise ValueError("Global median created an enclosed hole")
    except ValueError:
        cleaned = _smooth_without_merging(closed)
        before_count = _check_identity(base, cleaned)
        smoothing_mode = "per_component_remove_only_median5"
    if np.any(ndimage.binary_fill_holes(cleaned) & ~cleaned):
        raise ValueError("The cleaned mask acquired an enclosed hole")

    result = cleaned.copy()
    source_labels, _ = ndimage.label(cleaned, structure=_CONNECT8)
    seed_labels, _ = ndimage.label(high, structure=_CONNECT8)
    pairs, ambiguous = _two_core_candidates(source_labels, seed_labels)
    accepted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for component_id, pair_ids in pairs:
        component = source_labels == component_id
        try:
            rebuilt, details = _split_pair(component, seed_labels, pair_ids)
            ys, xs = np.nonzero(component)
            height, width = component.shape
            y0, y1 = max(0, int(ys.min()) - 12), min(height, int(ys.max()) + 13)
            x0, x1 = max(0, int(xs.min()) - 12), min(width, int(xs.max()) + 13)
            local_result = result[y0:y1, x0:x1]
            local_result[component[y0:y1, x0:x1]] = False
            local_result |= rebuilt
            accepted.append({"original_component_id": component_id, **details})
        except ValueError as exc:
            skipped.append({
                "original_component_id": component_id,
                "seed_component_ids_at_threshold_0p45": list(pair_ids),
                "reason": str(exc),
            })

    final_labels, final_count = ndimage.label(result, structure=_CONNECT8)
    if final_count != before_count + len(accepted):
        raise ValueError("Final component count differs from accepted splits")
    if final_count and np.min(np.bincount(final_labels.ravel())[1:]) < _MIN_AREA:
        raise ValueError("Final mask contains a component below 500 pixels")
    if np.any(ndimage.binary_fill_holes(result) & ~result):
        raise ValueError("Final mask contains an enclosed hole")
    if np.any(result & ~cleaned):
        raise ValueError("Separation added foreground outside the cleaned mask")
    if np.any(high & cleaned & ~result):
        raise ValueError("Separation removed a high-confidence pixel")
    changed = cleaned != result
    permitted = np.zeros_like(cleaned)
    for row in accepted:
        permitted |= source_labels == row["original_component_id"]
    if np.any(changed & ~permitted):
        raise ValueError("Separation changed pixels outside accepted pairs")

    report: dict[str, object] = {
        "shape": list(low.shape),
        "minimum_raw_component_area_pixels": _MIN_AREA,
        "closing_radius_pixels": _CLOSE_RADIUS,
        "smoothing_mode": smoothing_mode,
        "raw_components": int(raw_count),
        "retained_components": int(retained_count),
        "radius4_merge_groups_reverted": int(reverted_groups),
        "components_before_split": int(before_count),
        "components_after_split": int(final_count),
        "raw_foreground_pixels": int(low.sum()),
        "cleaned_foreground_pixels": int(cleaned.sum()),
        "final_foreground_pixels": int(result.sum()),
        "accepted_splits": accepted,
        "skipped_two_core_candidates": skipped,
        "ambiguous_multiple_cores": ambiguous,
        "high_core_provenance": "same-model binary prediction; not manual truth",
    }
    return result, report
