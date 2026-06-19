"""Shared CTF-BDI process-role identifiers used by tools and model adapters."""

PROCESS_STATES = (
    "background",
    "pre_start_transition",
    "early_action",
    "action_body",
    "pre_end_transition",
    "post_end_background",
    "uncertain",
)

DETAIL_ROLES = (
    "start_detail",
    "end_detail",
    "short_action_body",
    "neighbor_separator",
    "uncertainty_probe",
)
FUSION_ROLES = ("fallback_scaffold", *DETAIL_ROLES)
ROLE_TO_ID = {role: idx for idx, role in enumerate(FUSION_ROLES)}
ID_TO_ROLE = {idx: role for role, idx in ROLE_TO_ID.items()}
ROLE_TO_ROUND = {
    "fallback_scaffold": 0,
    "neighbor_separator": 1,
    "start_detail": 2,
    "end_detail": 2,
    "short_action_body": 3,
    "uncertainty_probe": 3,
}
ROLE_ID_TO_ROUND = {ROLE_TO_ID[role]: round_id for role, round_id in ROLE_TO_ROUND.items()}
NUM_CTF_BDI_ROLES = len(FUSION_ROLES)
NUM_CTF_BDI_ROUNDS = max(ROLE_TO_ROUND.values()) + 1
