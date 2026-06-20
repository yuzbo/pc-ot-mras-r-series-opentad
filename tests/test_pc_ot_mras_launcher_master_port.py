from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


LAUNCHERS = {
    "r16": (
        ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch",
        "31000",
    ),
    "r17": (
        ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r17_formal_train_n16r4.sbatch",
        "32000",
    ),
    "r18": (
        ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r18_aux_formal_train_n16r4.sbatch",
        "33000",
    ),
}


def test_pc_ot_mras_launchers_normalize_invalid_inherited_master_port():
    for path, base_port in LAUNCHERS.values():
        text = path.read_text(encoding="utf-8")

        assert f"DEFAULT_MASTER_PORT=$(({base_port} + (SLURM_JOB_ID_FOR_PORT % 20000)))" in text
        assert 'REQUESTED_MASTER_PORT="${MASTER_PORT:-}"' in text
        assert 'MASTER_PORT_SOURCE="environment"' in text
        assert "ignoring invalid inherited MASTER_PORT=$MASTER_PORT" in text
        assert "outside [1024,65535]" in text
        assert "normalized MASTER_PORT must be numeric" in text
        assert "normalized MASTER_PORT must be between 1024 and 65535" in text
        assert "master_port_source=$MASTER_PORT_SOURCE" in text
        assert "default_master_port=$DEFAULT_MASTER_PORT" in text


def test_pc_ot_mras_launcher_default_ports_stay_in_safe_range():
    for _, base_port in LAUNCHERS.values():
        base = int(base_port)
        min_port = base
        max_port = base + 19999

        assert 1024 <= min_port <= 65535
        assert 1024 <= max_port <= 65535
