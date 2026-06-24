from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bh_sdc_full_train_n16r4.sbatch"
FORBIDDEN_SHARED = "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730"


def test_bh_sdc_launcher_is_static_precheck_only_and_not_env_unlockable():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3" in text
    assert FORBIDDEN_SHARED not in text
    assert "#SBATCH --gpus=1" not in text
    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "ALLOW_BH_SDC_FULL_TRAIN" not in text
    assert "tools/train.py" not in text
    assert "tools/test.py" not in text
    assert "allowed_entrypoints" in text
    assert "exit 64" in text


def test_bh_sdc_launcher_shell_syntax_when_bash_is_available():
    bash = shutil.which("bash")
    if bash is None:
        return
    if "windows\\system32\\bash" in bash.lower():
        return
    subprocess.run([bash, "-n", str(LAUNCHER)], cwd=str(ROOT), check=True, timeout=30)
