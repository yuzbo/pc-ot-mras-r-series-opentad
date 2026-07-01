import json
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tools" / "bata" / "train_lowres_action_probe.py"
LOWRES_PROBE_SCRIPT = ROOT / "scripts" / "run_c3_lowres_action_probe_inside_pcot_dbg2g_v2_20260625.sh"
TCN_PROBE_GPU1_SCRIPT = ROOT / "scripts" / "run_c3_tcn_coarse_probe_gpu1_20260701.sh"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("train_lowres_action_probe_for_tests", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_torch(monkeypatch, probe, fake_torch):
    fake_nn = types.ModuleType("torch.nn")
    class FakeModule:
        def to(self, *args, **kwargs):
            return self

        def train(self):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

        def named_parameters(self):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return state_dict

    class FakeIdentity:
        def __call__(self, x):
            return x

    class FakeLinear:
        def __init__(self, in_features=None, out_features=None):
            self.in_features = in_features
            self.out_features = out_features

    class FakeLazyLinear(FakeLinear):
        pass

    fake_nn.Module = FakeModule
    fake_nn.Identity = FakeIdentity
    fake_nn.Linear = FakeLinear
    fake_nn.LazyLinear = FakeLazyLinear
    fake_torch.nn = fake_nn
    fake_torch_module = types.ModuleType("torch")
    fake_torch_module.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setattr(probe, "_import_torch", lambda: (fake_torch, None))
    monkeypatch.setattr(probe, "torch", fake_torch, raising=False)
    monkeypatch.setattr(probe, "nn", fake_nn, raising=False)


class _FakeTensor:
    def __init__(self, payload, *, ndim=0, device="cpu"):
        self.payload = payload
        self.ndim = ndim
        self.device = device

    @property
    def shape(self):
        if isinstance(self.payload, list):
            if self.payload and isinstance(self.payload[0], list):
                return (len(self.payload), len(self.payload[0]))
            return (len(self.payload),)
        return ()

    def to(self, device):
        return self

    def bool(self):
        return self

    def __iter__(self):
        return iter(self.payload)

    def __getitem__(self, idx):
        return self.payload[idx]

    def __len__(self):
        return len(self.payload)


class _FakeTorchModule:
    def __init__(self):
        self.nn = type(
            "FakeNN",
            (),
            {
                "Module": object,
                "Identity": lambda *args, **kwargs: None,
                "Linear": lambda *args, **kwargs: None,
                "LazyLinear": lambda *args, **kwargs: None,
            },
        )()

    def no_grad(self):
        return nullcontext()


class _FakeInput:
    def __init__(self, ndim=5, device="cpu"):
        self.ndim = ndim
        self.device = device

    def to(self, device):
        return self


def test_build_action_targets_marks_frames_inside_any_gt_segment():
    probe = load_probe_module()

    valid = [[True, True, True, True, True, True, False]]
    gt_segments = [[[1.0, 3.0], [4.2, 5.6]]]

    target = probe.build_action_targets(valid, gt_segments)

    assert target == [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]]


def test_c3_lowres_action_probe_wrapper_source_defaults_to_coarse_actionness_reader_for_action_logits():
    probe = load_probe_module()

    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert probe.DEFAULT_READER_TYPE == "PCOTMRASCoarseActionnessFrameScout"
    assert "PCOTMRASCoarseActionnessFrameScout" in source
    assert "build_selector(reader_cfg)" in source
    assert 'reader_outputs["action_logits"]' in source


def test_lowres_action_probe_rejects_unsupported_reader_instead_of_falling_back():
    probe = load_probe_module()

    cfg = types.SimpleNamespace(model={"frame_selector": {"reader": {"type": "UnknownReader"}}})

    with pytest.raises(ValueError, match="action probe expects"):
        probe._reader_cfg_from_config(cfg)


def test_binary_action_metrics_report_perfect_and_inverted_rankings():
    probe = load_probe_module()
    target = [[0.0, 1.0, 1.0, 0.0]]
    valid = [[True, True, True, True]]

    perfect = probe.compute_binary_action_metrics(
        logits=[[-4.0, 4.0, 3.0, -3.0]],
        target=target,
        valid=valid,
    )
    inverted = probe.compute_binary_action_metrics(
        logits=[[4.0, -4.0, -3.0, 3.0]],
        target=target,
        valid=valid,
    )

    assert perfect["roc_auc"] == 1.0
    assert perfect["average_precision"] == 1.0
    assert perfect["best_f1"] == 1.0
    assert perfect["accuracy"] == 1.0
    assert perfect["balanced_accuracy"] == 1.0
    assert inverted["roc_auc"] == 0.0
    assert inverted["average_precision"] < perfect["average_precision"]
    assert inverted["accuracy"] == 0.0
    assert inverted["balanced_accuracy"] == 0.0


def test_prepare_probe_inputs_keeps_c3_descriptors_and_mobilenet_images():
    probe = load_probe_module()
    probe.make_lowres_descriptors = lambda inputs, *, scout_spatial_size, normalize=True: (
        "c3",
        scout_spatial_size,
        normalize,
    )
    probe.make_lowres_frame_images = lambda inputs, *, spatial_size, normalize=False: (
        "mobilenet",
        spatial_size,
        normalize,
    )

    assert probe.prepare_probe_inputs("inputs", probe_model="c3-reader", spatial_size=4) == ("c3", 4, True)
    assert probe.prepare_probe_inputs("inputs", probe_model="mobilenetv3", spatial_size=4) == ("mobilenet", 4, False)
    assert probe.prepare_probe_inputs("inputs", probe_model="temporal-tcn", spatial_size=4) == ("mobilenet", 4, False)


def test_apply_dataset_overrides_updates_all_configured_splits():
    probe = load_probe_module()
    cfg = {
        "dataset": {
            "train": {"ann_file": "old_ann", "class_map": "old_map", "data_path": "old_train"},
            "val": {"ann_file": "old_ann", "class_map": "old_map", "data_path": "old_val"},
            "test": {"ann_file": "old_ann", "class_map": "old_map", "data_path": "old_test"},
        }
    }

    overrides = probe.apply_dataset_overrides(
        cfg,
        ann_file="/ann.json",
        class_map="/category_idx.txt",
        train_data_path="/train",
        val_data_path="/val",
        test_data_path="/test",
        train_subset_name="training",
        val_subset_name="training",
        test_subset_name="validation",
    )

    assert overrides == {
        "ann_file": "/ann.json",
        "class_map": "/category_idx.txt",
        "train_data_path": "/train",
        "val_data_path": "/val",
        "test_data_path": "/test",
        "train_subset_name": "training",
        "val_subset_name": "training",
        "test_subset_name": "validation",
    }
    assert cfg["dataset"]["train"]["ann_file"] == "/ann.json"
    assert cfg["dataset"]["val"]["class_map"] == "/category_idx.txt"
    assert cfg["dataset"]["train"]["data_path"] == "/train"
    assert cfg["dataset"]["val"]["data_path"] == "/val"
    assert cfg["dataset"]["test"]["data_path"] == "/test"
    assert cfg["dataset"]["val"]["subset_name"] == "training"


def test_apply_fast_lowres_pipeline_rewrites_video_pipeline_and_probe_window():
    probe = load_probe_module()
    cfg = {
        "dataset": {
            "train": {
                "pipeline": [
                    {"type": "PrepareVideoInfo", "format": "mp4"},
                    {"type": "mmaction.DecordInit", "num_threads": 4},
                    {"type": "LoadFrames", "method": "random_trunc", "trunc_len": 768},
                    {"type": "mmaction.DecordDecode"},
                    {"type": "mmaction.RandomResizedCrop"},
                    {"type": "Collect", "inputs": "imgs", "keys": ["masks", "gt_segments", "gt_labels"]},
                ]
            },
            "val": {
                "window_size": 768,
                "pipeline": [
                    {"type": "PrepareVideoInfo", "format": "mp4"},
                    {"type": "mmaction.DecordInit", "num_threads": 4},
                    {"type": "LoadFrames", "method": "sliding_window"},
                    {"type": "mmaction.DecordDecode"},
                    {"type": "mmaction.CenterCrop", "crop_size": 160},
                    {"type": "Collect", "inputs": "imgs", "keys": ["masks", "gt_segments", "gt_labels"]},
                ],
            },
        }
    }

    rewrites = probe.apply_fast_lowres_pipeline(cfg, spatial_size=32, probe_window_size=192)

    train_pipeline = cfg["dataset"]["train"]["pipeline"]
    val_pipeline = cfg["dataset"]["val"]["pipeline"]
    assert rewrites == {"train": "fast_lowres_32", "val": "fast_lowres_32"}
    assert train_pipeline[2]["trunc_len"] == 192
    assert train_pipeline[4] == {"type": "mmaction.Resize", "scale": (32, 32), "keep_ratio": False}
    assert train_pipeline[-1]["keys"] == ["masks", "gt_segments", "gt_labels"]
    assert cfg["dataset"]["val"]["window_size"] == 192
    assert val_pipeline[4] == {"type": "mmaction.Resize", "scale": (32, 32), "keep_ratio": False}


def test_mobilenetv3_probe_uses_pretrained_cnn_and_outputs_frame_logits(monkeypatch):
    probe = load_probe_module()

    class FakeWeights:
        DEFAULT = object()

    captured = {}

    class FakeBackbone:
        def __init__(self):
            self.classifier = ["dropout", "linear"]

        def __call__(self, x):
            import torch

            return torch.ones((x.shape[0], 1000), dtype=x.dtype, device=x.device)

    def fake_mobilenet_v3_small(*, weights):
        captured["weights"] = weights
        return FakeBackbone()

    monkeypatch.setattr(
        probe,
        "_import_torchvision_mobilenet",
        lambda: (fake_mobilenet_v3_small, FakeWeights),
    )
    fake_torch = type(
        "FakeTorch",
        (),
        {
            "tensor": lambda *args, **kwargs: args[0],
            "float32": "float32",
        },
    )()

    class FakeNN:
        class Module:
            def __init__(self):
                self._params = []

            def to(self, *args, **kwargs):
                return self

            def train(self):
                return self

            def eval(self):
                return self

            def parameters(self):
                return []

            def named_parameters(self):
                return []

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                return state_dict

        class Identity:
            def __call__(self, x):
                return x

        class Linear:
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features

        class LazyLinear(Linear):
            def __init__(self, out_features):
                self.in_features = None
                self.out_features = out_features

    _install_fake_torch(monkeypatch, probe, fake_torch)

    model = probe.C3MobileNetV3ActionProbe(pretrained=True, variant="small")
    assert captured["weights"] is FakeWeights.DEFAULT
    assert model.backbone.__class__.__name__ == "FakeBackbone"


def test_sampling_quality_metrics_report_boundary_coverage_and_gap():
    probe = load_probe_module()

    metrics = probe.compute_sampling_quality_from_logits(
        logits=[[0.0, 4.0, 3.0, -2.0, 2.5, -3.0]],
        target=[[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
        valid=[[True, True, True, True, True, True]],
        gt_segments=[[[1.0, 2.5], [4.0, 4.8]]],
        budget=3,
        boundary_radius=1,
    )

    assert metrics["budget"] == 3
    assert metrics["sample_count"] == 3
    assert metrics["selected_indices"] == [[1, 2, 4]]
    assert metrics["action_selected_fraction"] == 1.0
    assert metrics["boundary_support_r1"] == 1.0
    assert metrics["max_gap"] == 2
    assert metrics["selected_run_count_mean"] == 2.0
    assert metrics["selected_run_count_p95"] == 2.0
    assert metrics["longest_selected_run_mean"] == 2.0
    assert metrics["longest_selected_run_p95"] == 2.0
    assert metrics["mean_selected_run_length"] == 1.5
    assert metrics["selected_run_count_by_window"] == [2]
    assert metrics["longest_selected_run_by_window"] == [2]
    assert metrics["selected_run_lengths_by_window"] == [[2, 1]]


def test_indirect_boundary_support_counts_each_gt_boundary_once():
    probe = load_probe_module()

    payload = probe.compute_indirect_selection_quality_from_logits(
        logits=[[4.0, 3.9, 3.8, -6.0, -6.0, -6.0]],
        target=[[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
        valid=[[True, True, True, True, True, True]],
        gt_segments=[[[2.0, 5.0]]],
        sample_ids=["duplicate_near_boundary"],
        budget=3,
        boundary_radius=1,
    )

    row = payload["per_sample_rows"][0]
    selected = row["selected_positions"]
    expected_hits = probe._boundary_hit_count(selected, [2.0, 5.0], radius=1)
    expected_support = expected_hits / 2.0
    assert expected_support <= 1.0
    assert payload["indirect"]["boundary_support_r1"] == expected_support
    assert row["boundary_support_r1"] == expected_support


def test_sampling_quality_run_metrics_are_stable_for_empty_selection():
    probe = load_probe_module()

    metrics = probe.compute_sampling_quality_from_logits(
        logits=[[1.0, 0.5, -1.0]],
        target=[[0.0, 1.0, 0.0]],
        valid=[[True, True, True]],
        gt_segments=[[]],
        budget=0,
        boundary_radius=1,
    )

    assert metrics["selected_indices"] == [[]]
    assert metrics["selected_run_count_mean"] == 0.0
    assert metrics["selected_run_count_p95"] == 0.0
    assert metrics["longest_selected_run_mean"] == 0.0
    assert metrics["longest_selected_run_p95"] == 0.0
    assert metrics["mean_selected_run_length"] is None
    assert metrics["selected_run_count_by_window"] == [0]
    assert metrics["longest_selected_run_by_window"] == [0]
    assert metrics["selected_run_lengths_by_window"] == [[]]


def test_sample_id_resolution_prefers_batch_ids_then_video_name_then_fallback():
    probe = load_probe_module()

    assert probe._resolve_sample_ids({"sample_ids": ["sid_a", "sid_b"]}, batch_idx=3, batch_size=2) == [
        "sid_a",
        "sid_b",
    ]
    assert probe._resolve_sample_ids({"video_name": "clip_01"}, batch_idx=7, batch_size=1) == ["clip_01"]
    assert probe._resolve_sample_ids({"metas": [{"video_name": "meta_clip"}]}, batch_idx=9, batch_size=1) == ["meta_clip"]
    assert probe._resolve_sample_ids({}, batch_idx=11, batch_size=2) == [
        "batch_00011|sample_00000",
        "batch_00011|sample_00001",
    ]


def test_indirect_selection_quality_serializes_sample_rows_with_stable_schema():
    probe = load_probe_module()

    payload = probe.compute_indirect_selection_quality_from_logits(
        logits=[[0.0, 4.0, 3.0, -2.0, 2.5, -3.0]],
        target=[[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
        valid=[[True, True, True, True, True, True]],
        gt_segments=[[[1.0, 2.5], [4.0, 4.8]]],
        sample_ids=["video_0001"],
        budget=3,
        boundary_radius=1,
    )

    assert payload["indirect"]["selected_role_counts"]["mixed_fill"] >= 0
    row = payload["per_sample_rows"][0]
    assert row["sample_id"] == "video_0001"
    assert row["selected_positions"]
    assert row["frame_signals"]["p_action"]
    assert isinstance(row["frame_signals"]["mixed_fill"][0], bool)
    assert row["selected_role_details"][0]["candidate_roles"]
    json.dumps(row)


def test_evaluate_aggregates_gt_segments_for_sampling_quality_on_both_probe_paths():
    probe = load_probe_module()
    batch = {
        "inputs": _FakeInput(ndim=5),
        "masks": _FakeTensor([[True, True, True, True, True, True]], ndim=2),
        "gt_segments": [[[1.0, 2.5], [4.0, 4.8]]],
        "sample_ids": ["video_0001"],
    }

    class DummyModel:
        def __init__(self, expected_ndim):
            self.expected_ndim = expected_ndim

        def eval(self):
            return self

        def __call__(self, inputs, valid):
            return [[0.0, 4.0, 3.0, -2.0, 2.5, -3.0]]

    for probe_model, expected_ndim in (("c3-reader", 3), ("mobilenetv3", 5)):
        probe._batch_inputs = lambda batch_arg: batch_arg["inputs"]
        probe.prepare_probe_inputs = lambda inputs, *, probe_model, spatial_size: _FakeInput(ndim=expected_ndim)
        probe._targets_to_torch = lambda valid, gt_segments, *, device: [[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]
        probe._import_torch = lambda: (type("FakeTorch", (), {"no_grad": lambda self=None: nullcontext()})(), None)
        metrics = probe.evaluate(
            model=DummyModel(expected_ndim),
            dataloader=[batch],
            device="cpu",
            scout_spatial_size=4,
            probe_model=probe_model,
            max_batches=1,
            epoch=1,
            total_epochs=1,
            progress_path=None,
            log_every_batches=0,
            coverage_budget_fraction=0.5,
            coverage_budget=3,
            boundary_radius=1,
        )

        assert metrics["sampling_quality"]["selected_indices"] == [[1, 2, 4]]
        assert metrics["sampling_quality"]["boundary_support_r1"] == 1.0


def test_evaluate_writes_indirect_selection_jsonl_with_sample_ids(tmp_path):
    probe = load_probe_module()
    batch = {
        "inputs": _FakeInput(ndim=5),
        "masks": _FakeTensor([[True, True, True, True, True, True]], ndim=2),
        "gt_segments": [[[1.0, 2.5], [4.0, 4.8]]],
        "video_name": ["video_alpha"],
    }

    class DummyModel:
        def eval(self):
            return self

        def __call__(self, inputs, valid):
            return [[0.0, 4.0, 3.0, -2.0, 2.5, -3.0]]

    sample_jsonl = tmp_path / "samples.jsonl"
    progress_jsonl = tmp_path / "progress.jsonl"
    probe._batch_inputs = lambda batch_arg: batch_arg["inputs"]
    probe.prepare_probe_inputs = lambda inputs, *, probe_model, spatial_size: _FakeInput(ndim=5)
    probe._targets_to_torch = lambda valid, gt_segments, *, device: [[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]
    probe._import_torch = lambda: (type("FakeTorch", (), {"no_grad": lambda self=None: nullcontext()})(), None)
    metrics = probe.evaluate(
        model=DummyModel(),
        dataloader=[batch],
        device="cpu",
        scout_spatial_size=4,
        probe_model="mobilenetv3",
        max_batches=1,
        epoch=1,
        total_epochs=1,
        progress_path=progress_jsonl,
        log_every_batches=0,
        coverage_budget_fraction=0.5,
        coverage_budget=3,
        boundary_radius=1,
        sample_jsonl_path=sample_jsonl,
    )

    assert metrics["sampling_quality"]["sample_count"] == 3
    rows = [json.loads(line) for line in sample_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert metrics["indirect_selection_quality"]["sample_count"] == 1
    assert "per_sample" not in metrics["indirect_selection_quality"]
    assert "per_sample_rows" not in metrics["indirect_selection_quality"]
    progress_rows = [json.loads(line) for line in progress_jsonl.read_text(encoding="utf-8").splitlines()]
    progress_payload = "\n".join(json.dumps(item) for item in progress_rows)
    assert "per_sample" not in progress_payload
    assert "per_sample_rows" not in progress_payload
    assert '"selected_indices":' not in progress_payload
    row = rows[0]
    assert row["sample_id"] == "video_alpha"
    assert row["probe_model"] == "mobilenetv3"
    assert row["spatial_size"] == 4
    assert row["selected_positions"]
    assert "p_action" in row and "entropy" in row and "mixed_fill" in row
    assert "frame_signals" in row
    assert isinstance(row["mixed_fill"][0], bool)


def test_parse_args_supports_mobilenetv3_32_64_probe_without_detector_path():
    probe = load_probe_module()

    args = probe.parse_args(
        [
            "--probe-model",
            "mobilenetv3",
            "--mobilenet-sizes",
            "32",
            "64",
            "--coverage-only",
        ]
    )

    assert args.probe_model == "mobilenetv3"
    assert args.mobilenet_sizes == [32, 64]
    assert args.coverage_only is True
    assert "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap" in args.config
    assert args.max_train_batches == 50
    assert args.max_val_batches == 50
    assert not hasattr(args, "detector_checkpoint")


def test_parse_args_supports_temporal_tcn_variants_and_rejects_unknown_variant():
    probe = load_probe_module()

    args = probe.parse_args(
        [
            "--probe-model",
            "temporal-tcn",
            "--scout-spatial-size",
            "64",
            "--tcn-variants",
            "lite",
            "dilated",
            "multiscale",
            "motion",
        ]
    )

    assert args.probe_model == "temporal-tcn"
    assert args.scout_spatial_size == 64
    assert args.tcn_variants == ["lite", "dilated", "multiscale", "motion"]

    with pytest.raises(SystemExit):
        probe.parse_args(["--probe-model", "temporal-tcn", "--tcn-variants", "unknown"])


def test_parse_args_accepts_zero_batch_caps_as_explicit_unlimited_probe_mode():
    probe = load_probe_module()

    args = probe.parse_args(["--max-train-batches", "0", "--max-val-batches", "0"])

    assert args.max_train_batches == 0
    assert args.max_val_batches == 0


def test_lowres_probe_v2_launcher_uses_c3_a_config_and_positive_batch_caps():
    text = LOWRES_PROBE_SCRIPT.read_text(encoding="utf-8")

    assert "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap" in text
    assert "--mobilenet-sizes 32 64" in text
    assert "--max-train-batches 50" in text
    assert "--max-val-batches 20" in text
    assert "--max-train-batches 0" not in text
    assert "--max-val-batches 0" not in text


def test_tcn_probe_gpu1_launcher_fail_closes_and_runs_all_variants():
    text = TCN_PROBE_GPU1_SCRIPT.read_text(encoding="utf-8")

    assert "OpenTAD_C3TCNCoarseProbe_20260701" in text
    assert "OpenTAD_Back_clean_20260629_588b272" not in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert "--probe-model temporal-tcn" in text
    assert "--scout-spatial-size 64" in text
    assert "--tcn-variants lite dilated multiscale motion" in text
    assert "--mobilenet-sizes" not in text
    assert "SLURM_STEP_GPUS" in text


def test_parse_args_exposes_seed_for_reproducible_probe_runs():
    probe = load_probe_module()

    args = probe.parse_args(["--seed", "7", "--coverage-only"])

    assert args.seed == 7


def test_parse_args_supports_probe_checkpoint_for_coverage_export():
    probe = load_probe_module()

    args = probe.parse_args(["--coverage-only", "--probe-checkpoint", "probe_reader.pth"])

    assert args.coverage_only is True
    assert args.probe_checkpoint == "probe_reader.pth"


def test_load_probe_checkpoint_calls_full_probe_load_state_dict(monkeypatch):
    probe = load_probe_module()
    loaded_state = {"backbone.classifier.3.weight": [1.0]}
    monkeypatch.setattr(probe, "_load_torch_state_dict", lambda path: loaded_state)

    class DummyProbe:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state_dict):
            self.loaded = state_dict
            return "ok"

    model = DummyProbe()
    result = probe._load_probe_checkpoint(model, "probe_reader.pth")

    assert result == "ok"
    assert model.loaded is loaded_state


def test_multisize_summary_directory_layout_is_size_specific():
    probe = load_probe_module()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mobilenet_path = probe._probe_out_dir(root, probe_model="mobilenetv3", spatial_size=32, multi_size=True)
        mobilenet_single = probe._probe_out_dir(root, probe_model="mobilenetv3", spatial_size=32, multi_size=False)
        c3_path = probe._probe_out_dir(root, probe_model="c3-reader", spatial_size=32, multi_size=True)
        tcn_path = probe._probe_out_dir(
            root,
            probe_model="temporal-tcn",
            spatial_size=64,
            multi_size=False,
            tcn_variant="lite",
            multi_variant=True,
        )

        assert mobilenet_path == root / "mobilenetv3_32"
        assert mobilenet_single == root
        assert c3_path == root
        assert tcn_path == root / "temporal_tcn_lite_64"


def test_build_probe_model_supports_temporal_tcn_branch(monkeypatch):
    probe = load_probe_module()
    captured = {}

    class FakeTemporalTCN:
        def __init__(self, *, variant, spatial_size):
            captured["variant"] = variant
            captured["spatial_size"] = spatial_size

    monkeypatch.setattr(probe, "C3TemporalTCNActionProbe", FakeTemporalTCN)
    args = probe.parse_args(["--probe-model", "temporal-tcn", "--tcn-variants", "dilated"])
    args.tcn_variant = "dilated"

    model, reader_cfg = probe._build_probe_model(args, cfg=types.SimpleNamespace(), spatial_size=64)

    assert isinstance(model, FakeTemporalTCN)
    assert reader_cfg is None
    assert captured == {"variant": "dilated", "spatial_size": 64}


def test_multisize_mobilenet_summary_exposes_per_size_results():
    probe = load_probe_module()

    summaries = [
        {
            "spatial_size": 32,
            "final_val": {"average_precision": 0.5, "roc_auc": 0.6},
            "out_dir": "root/mobilenetv3_32",
        },
        {
            "spatial_size": 64,
            "final_val": {"average_precision": 0.7, "roc_auc": 0.8},
            "out_dir": "root/mobilenetv3_64",
        },
    ]
    combined = probe._combine_multisize_summaries(
        base_summary={"probe_model": "mobilenetv3"},
        summaries=summaries,
        args_out_dir=Path("root"),
    )

    assert combined["schema_version"] == "lowres_action_probe_multisize_v1"
    assert combined["mobilenetv3_32"]["out_dir"] == "root/mobilenetv3_32"
    assert combined["mobilenetv3_64"]["out_dir"] == "root/mobilenetv3_64"
    assert combined["comparison"]["average_precision_delta_64_minus_32"] == pytest.approx(0.2)


def test_tcn_variant_summary_exposes_per_variant_results():
    probe = load_probe_module()

    summaries = [
        {
            "probe_model": "temporal-tcn",
            "tcn_variant": "lite",
            "spatial_size": 64,
            "final_val": {"average_precision": 0.55, "roc_auc": 0.61},
            "out_dir": "root/temporal_tcn_lite_64",
        },
        {
            "probe_model": "temporal-tcn",
            "tcn_variant": "motion",
            "spatial_size": 64,
            "final_val": {"average_precision": 0.66, "roc_auc": 0.72},
            "out_dir": "root/temporal_tcn_motion_64",
        },
    ]

    combined = probe._combine_tcn_variant_summaries(
        base_summary={"probe_model": "temporal-tcn", "seed": 3},
        summaries=summaries,
        args_out_dir=Path("root"),
    )

    assert combined["schema_version"] == "lowres_action_probe_tcn_variants_v1"
    assert combined["probe_model"] == "temporal-tcn"
    assert combined["tcn_variants"] == ["lite", "motion"]
    assert combined["temporal_tcn_lite"]["out_dir"] == "root/temporal_tcn_lite_64"
    assert combined["temporal_tcn_motion"]["out_dir"] == "root/temporal_tcn_motion_64"
    assert combined["comparison"]["best_average_precision_variant"] == "motion"
    assert combined["comparison"]["average_precision_by_variant"]["lite"] == 0.55


def test_load_torch_state_dict_accepts_probe_state_dict(tmp_path):
    probe = load_probe_module()
    import torch

    path = tmp_path / "probe_reader.pth"
    torch.save({"probe_state_dict": {"backbone.weight": torch.tensor([1.0])}}, path)

    loaded = probe._load_torch_state_dict(str(path))
    assert "backbone.weight" in loaded
    assert torch.equal(loaded["backbone.weight"], torch.tensor([1.0]))
