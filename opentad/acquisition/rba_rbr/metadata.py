def resolve_backbone_time_axis_meta(meta):
    """Return raw-frame time-axis metadata for backbone time embeddings.

    Detector/head geometry may use generic `irregular_selected_positions`
    populated with feature centers. The backbone sees raw frames, so RBA-RBR
    must prefer its raw selected positions when present.
    """

    if "rba_rbr_raw_selected_positions" in meta:
        return (
            meta.get("rba_rbr_raw_selected_positions", []),
            meta.get("rba_rbr_raw_selected_valid_len", None),
            "rba_rbr_raw",
        )
    if "bvr_twb_raw_selected_positions" in meta:
        return (
            meta.get("bvr_twb_raw_selected_positions", []),
            meta.get("bvr_twb_raw_selected_valid_len", None),
            "bvr_twb_raw",
        )
    return (
        meta.get("irregular_selected_positions", []),
        meta.get("irregular_selected_valid_len", None),
        "generic_irregular",
    )


def has_backbone_time_axis_meta(meta):
    return (
        ("irregular_selected_positions" in meta and "irregular_selected_valid_len" in meta)
        or ("bvr_twb_raw_selected_positions" in meta and "bvr_twb_raw_selected_valid_len" in meta)
        or ("rba_rbr_raw_selected_positions" in meta and "rba_rbr_raw_selected_valid_len" in meta)
    )
