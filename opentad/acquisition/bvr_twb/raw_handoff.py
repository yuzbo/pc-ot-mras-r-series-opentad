from .types import sorted_unique_positions


def build_sparse_raw_handoff(selected_positions, dense_T, pad_to=None):
    positions = sorted_unique_positions(selected_positions, dense_T)
    if not positions:
        raise ValueError("raw handoff requires at least one selected position")

    raw_frame_inds = list(positions)
    padded_duplicate_count = 0
    if pad_to is not None:
        pad_to = int(pad_to)
        if pad_to < len(raw_frame_inds):
            raise ValueError("pad_to cannot be smaller than selected raw frame count")
        padded_duplicate_count = pad_to - len(raw_frame_inds)
        if padded_duplicate_count:
            raw_frame_inds.extend([raw_frame_inds[-1]] * padded_duplicate_count)

    return {
        "raw_frame_inds_in": [int(pos) for pos in raw_frame_inds],
        "decoded_frame_count": int(len(raw_frame_inds)),
        "decoded_unique_count": int(len(set(raw_frame_inds))),
        "padded_duplicate_count": int(padded_duplicate_count),
    }
