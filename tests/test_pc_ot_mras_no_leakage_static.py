import io
import tokenize
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SOURCES = [
    ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
    ROOT / "opentad" / "models" / "necks" / "pc_ot_mras_detector_bridge.py",
    ROOT / "tools" / "bata" / "export_pc_ot_mras_hard_positions.py",
]
FORBIDDEN_TOKENS = {
    "gt",
    "ground_truth",
    "oracle",
    "teacher",
    "cache",
    "raw_prediction",
    "raw_predictions",
    "prediction_cache",
    "bca",
    "bce",
}


def _code_tokens_without_comments_or_strings(path: Path) -> List[str]:
    source = path.read_text(encoding="utf-8")
    tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in {tokenize.COMMENT, tokenize.STRING, tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE}:
            continue
        tokens.append(tok.string.lower())
    return tokens


def test_pc_ot_mras_deploy_sources_do_not_reference_forbidden_shortcuts():
    missing = [str(path) for path in DEPLOY_SOURCES if not path.exists()]
    assert not missing, missing
    for path in DEPLOY_SOURCES:
        tokens = _code_tokens_without_comments_or_strings(path)
        joined = " ".join(tokens)
        for forbidden in FORBIDDEN_TOKENS:
            assert forbidden not in tokens
            assert forbidden not in joined
