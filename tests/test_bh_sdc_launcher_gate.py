from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bh_sdc_full_train_n16r4.sbatch"
FORBIDDEN_SHARED = "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730"


def test_bh_sdc_launcher_is_payload_gated_n16r4_train_candidate():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3" in text
    assert FORBIDDEN_SHARED not in text
    assert "#SBATCH -p gpu" in text
    assert "#SBATCH --gpus=1" in text
    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "ALLOW_BH_SDC_FULL_TRAIN" in text
    assert "ALLOW_LOGIN_NODE_DEBUG" in text
    assert "OPENTAD_BH_SDC_GATE_JSON" in text
    assert "OPENTAD_BH_SDC_GATE_SHA256" in text
    assert "validate_bh_sdc_full_train_gate.py" in text
    assert "ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text
    assert "REMOTE_SYNC_TO_N16R4:~/run/yuzibo/OpenTAD_Back_check" in text
    assert "exit 64" in text


def test_bh_sdc_launcher_refuses_training_without_gate_payload_when_bash_is_available():
    bash = shutil.which("bash")
    if bash is None:
        return
    if "windows\\system32\\bash" in bash.lower():
        return

    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": ".",
            "PRECHECK_ONLY": "0",
            "ALLOW_BH_SDC_FULL_TRAIN": "1",
            "ALLOW_LOGIN_NODE_DEBUG": "1",
            "BH_SDC_SKIP_MODULES": "1",
        }
    )
    result = subprocess.run(
        [bash, str(LAUNCHER)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "OPENTAD_BH_SDC_GATE_JSON" in result.stdout
    assert "refusing" in result.stdout.lower() or "requires" in result.stdout.lower()


def test_bh_sdc_launcher_shell_syntax_when_bash_is_available():
    bash = shutil.which("bash")
    if bash is None:
        return
    if "windows\\system32\\bash" in bash.lower():
        return
    subprocess.run([bash, "-n", str(LAUNCHER)], cwd=str(ROOT), check=True, timeout=30)
