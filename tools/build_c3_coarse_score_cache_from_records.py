import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


ROUTE_LABELS = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]


def _load_record(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_records(input_path):
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif input_path.is_dir():
        for path in sorted(input_path.glob("*.npz")):
            yield _load_record(path)
    else:
        yield _load_record(input_path)


def _as_score_array(record):
    if "action_score" in record:
        return "action_score", np.asarray(record["action_score"], dtype=np.float32).reshape(-1)
    if "action_scores" in record:
        return "action_score", np.asarray(record["action_scores"], dtype=np.float32).reshape(-1)
    if "action_logit" in record:
        return "action_logit", np.asarray(record["action_logit"], dtype=np.float32).reshape(-1)
    if "action_logits" in record:
        return "action_logit", np.asarray(record["action_logits"], dtype=np.float32).reshape(-1)
    raise AssertionError("record requires action_score/action_scores or action_logit/action_logits")


def build_cache(input_path, output_dir, score_source, axis="global_snippet_index", overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = {}
    for record in _iter_records(input_path):
        video_name = record.get("video_name")
        if isinstance(video_name, np.ndarray):
            video_name = str(video_name.tolist())
        if not video_name:
            raise AssertionError("each coarse score record requires video_name")
        key, values = _as_score_array(record)
        out_name = f"{video_name}.npz"
        out_path = output_dir / out_name
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing score file: {out_path}")
        np.savez(out_path, **{key: values, "video_name": video_name, "axis": axis})
        videos[video_name] = {"file": out_name, "num_frames": int(values.size)}

    manifest = {
        "schema_version": 1,
        "route_labels": ROUTE_LABELS,
        "score_source": score_source,
        "uses_gt": False,
        "axis": axis,
        "videos": videos,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build fail-closed C3 coarse action score cache from JSONL/NPZ records.")
    parser.add_argument("--input", required=True, help="JSONL, NPZ, or directory of NPZ records")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-source", required=True)
    parser.add_argument("--axis", default="global_snippet_index")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_cache(args.input, args.output_dir, args.score_source, axis=args.axis, overwrite=args.overwrite)
    print(json.dumps({"status": "ok", "videos": len(manifest["videos"]), "manifest": str(Path(args.output_dir) / "manifest.json")}))


if __name__ == "__main__":
    main()
