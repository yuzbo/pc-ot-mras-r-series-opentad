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
MATRIX_IMAGE_GPU1_SCRIPT = ROOT / "scripts" / "run_c3_matrix_zoo_image_backbone_probe_gpu1_20260701.sh"
MATRIX_VIDEO_GPU1_SCRIPT = ROOT / "scripts" / "run_c3_matrix_zoo_video_probe_gpu1_20260701.sh"


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
    assert probe.prepare_probe_inputs("inputs", probe_model="matrix-zoo", spatial_size=4) == ("mobilenet", 4, False)


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


def test_sample_id_resolution_preserves_window_key_from_metas():
    probe = load_probe_module()

    assert probe._resolve_sample_ids(
        {
            "video_name": ["plain_video"],
            "metas": [{"video_name": "video_test_0001", "window_start_frame": 768.0}],
        },
        batch_idx=2,
        batch_size=1,
    ) == ["video_test_0001|768"]


def test_eval_export_window_overrides_make_probe_ledger_detector_grid_aligned():
    probe = load_probe_module()
    cfg = {
        "dataset": {
            "val": {
                "window_overlap_ratio": 0.25,
                "ioa_thresh": 0.75,
                "filter_gt": True,
            }
        }
    }

    applied = probe.apply_eval_export_window_overrides(
        cfg,
        eval_window_overlap_ratio=0.5,
        eval_include_all_windows=True,
    )

    assert applied == {
        "val_window_overlap_ratio": 0.5,
        "val_ioa_thresh": 0,
        "val_filter_gt": False,
        "val_include_all_windows": True,
    }
    assert cfg["dataset"]["val"]["window_overlap_ratio"] == 0.5
    assert cfg["dataset"]["val"]["ioa_thresh"] == 0
    assert cfg["dataset"]["val"]["filter_gt"] is False


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


def test_indirect_selection_quality_reports_strategy_comparison():
    probe = load_probe_module()

    payload = probe.compute_indirect_selection_quality_from_logits(
        logits=[[-6.0, -5.0, 6.0, 6.0, 6.0, -5.0, -6.0]],
        target=[[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        valid=[[True, True, True, True, True, True, True]],
        gt_segments=[[[2.0, 5.0]]],
        sample_ids=["state_change_window"],
        budget=2,
        boundary_radius=0,
    )

    strategies = payload["strategy_metrics"]
    assert "topk_action_logit" in strategies
    assert "delta_p_action" in strategies
    assert "entropy_uncertainty" in strategies
    assert "boundary_score" in strategies
    assert "weighted_transition_mix" in strategies
    assert "state_machine_mix" in strategies
    assert payload["strategy_comparison"]["best_boundary_support_strategy"] in strategies
    assert strategies["delta_p_action"]["boundary_support_r0"] == 1.0
    assert strategies["topk_action_logit"]["boundary_support_r0"] < strategies["delta_p_action"]["boundary_support_r0"]
    row = payload["per_sample_rows"][0]
    assert row["strategy_selected_positions"]["delta_p_action"] == [2, 5]
    assert row["selected_sources"]["strategies"] == list(strategies)


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
            "--eval-window-overlap-ratio",
            "0.5",
            "--eval-include-all-windows",
        ]
    )

    assert args.probe_model == "mobilenetv3"
    assert args.mobilenet_sizes == [32, 64]
    assert args.coverage_only is True
    assert args.eval_window_overlap_ratio == 0.5
    assert args.eval_include_all_windows is True
    assert "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap" in args.config
    assert args.max_train_batches == 50
    assert args.max_val_batches == 50
    assert not hasattr(args, "detector_checkpoint")


def test_parse_args_supports_temporal_tcn_variants_and_rejects_unknown_variant():
    probe = load_probe_module()

    expected_variants = [
        "lite",
        "dilated",
        "multiscale",
        "motion",
        "residual",
        "gated",
        "separable_dilated",
        "causal_dilated",
        "ms_tcnpp",
        "c2f_tcn",
        "asformer_lite",
        "fact_lite",
        "temporal_mamba_lite",
    ]
    args = probe.parse_args(
        [
            "--probe-model",
            "temporal-tcn",
            "--scout-spatial-size",
            "64",
            "--tcn-variants",
            *expected_variants,
        ]
    )

    assert args.probe_model == "temporal-tcn"
    assert args.scout_spatial_size == 64
    assert tuple(expected_variants) == probe.SUPPORTED_TCN_VARIANTS
    assert args.tcn_variants == expected_variants

    with pytest.raises(SystemExit):
        probe.parse_args(["--probe-model", "temporal-tcn", "--tcn-variants", "unknown"])


def test_parse_args_supports_official_action_segmentation_backends():
    probe = load_probe_module()

    expected = [
        "official_ms_tcn2",
        "official_asformer",
        "official_fact",
        "official_video_mamba_asformer",
    ]
    args = probe.parse_args(
        [
            "--probe-model",
            "official-action-seg",
            "--official-action-seg-backends",
            *expected,
            "--scout-spatial-size",
            "64",
        ]
    )

    assert args.probe_model == "official-action-seg"
    assert tuple(expected) == probe.SUPPORTED_OFFICIAL_ACTION_SEG_BACKENDS
    assert args.official_action_seg_backends == expected

    with pytest.raises(SystemExit):
        probe.parse_args(["--probe-model", "official-action-seg", "--official-action-seg-backends", "asformer_lite"])


def test_official_action_seg_probe_forwards_binary_logits_with_source_metadata():
    probe = load_probe_module()
    torch = pytest.importorskip("torch")

    frames = torch.rand(2, 9, 3, 16, 16)
    valid = torch.ones(2, 9, dtype=torch.bool)

    for backend in ("official_ms_tcn2", "official_asformer", "official_fact"):
        reader = probe.C3OfficialActionSegmentationProbe(
            backend=backend,
            spatial_size=16,
            hidden_dim=16,
            num_layers=1,
        )
        logits = reader(frames, valid)

        assert tuple(logits.shape) == (2, 9)
        assert torch.isfinite(logits).all()
        assert reader.official_source["backend"] == backend
        assert reader.official_source["repo_path"]
        assert "lite" not in reader.official_source["backend"]


def test_official_action_seg_probe_uses_action_minus_background_logit_and_masks_invalid_frames():
    probe = load_probe_module()
    torch = pytest.importorskip("torch")

    class ConstantStem(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channels = int(channels)

        def forward(self, frames):
            return torch.ones(frames.shape[0], self.channels, 1, 1, device=frames.device)

    class FakeTwoClassTemporal(torch.nn.Module):
        def forward(self, features, mask=None):
            batch, _channels, dense_len = features.shape
            background = torch.full((batch, dense_len), -2.0, device=features.device)
            action = torch.full((batch, dense_len), 3.0, device=features.device)
            return torch.stack([torch.stack([background, action], dim=1)], dim=0)

    class FakeFactTemporal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_list = [types.SimpleNamespace(frame_clogit=None)]

        def _forward_one_video(self, seq):
            dense_len = seq.shape[0]
            background = torch.full((dense_len,), -2.0, device=seq.device)
            action = torch.full((dense_len,), 3.0, device=seq.device)
            self.block_list[-1].frame_clogit = torch.stack([background, action], dim=1).unsqueeze(1)

    frames = torch.rand(2, 5, 3, 16, 16)
    valid = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
    expected = torch.where(valid, torch.full_like(valid.float(), 5.0), torch.zeros_like(valid.float()))

    for backend in ("official_ms_tcn2", "official_asformer", "official_fact"):
        reader = probe.C3OfficialActionSegmentationProbe(
            backend=backend,
            spatial_size=16,
            hidden_dim=16,
            num_layers=1,
        )
        reader.spatial_stem = ConstantStem(reader.hidden_dim)
        if backend == "official_fact":
            reader.official_temporal = FakeFactTemporal()
        else:
            reader.official_temporal = FakeTwoClassTemporal()
        reader.module.official_temporal = reader.official_temporal

        logits = reader(frames, valid)

        assert torch.allclose(logits, expected)


def test_official_video_mamba_backend_fails_closed_without_dependency():
    probe = load_probe_module()

    if probe.official_action_seg_backend_available("official_video_mamba_asformer"):
        pytest.skip("mamba_ssm is installed in this environment; fail-closed path is not exercised")

    with pytest.raises(RuntimeError, match="mamba_ssm"):
        probe.C3OfficialActionSegmentationProbe(backend="official_video_mamba_asformer", spatial_size=16)


def test_parse_args_supports_matrix_zoo_model_ids_and_defaults_from_matrix():
    probe = load_probe_module()

    args = probe.parse_args(
        [
            "--probe-model",
            "matrix-zoo",
            "--matrix-model-ids",
            "timm_resnet18_tcn",
            "torchvision_r3d_18",
            "--scout-spatial-size",
            "64",
            "--matrix-video-clip-len",
            "8",
            "--matrix-video-anchor-stride",
            "4",
            "--matrix-continue-on-model-error",
        ]
    )

    assert args.probe_model == "matrix-zoo"
    assert args.matrix_model_ids == ["timm_resnet18_tcn", "torchvision_r3d_18"]
    assert args.matrix_video_clip_len == 8
    assert args.matrix_video_anchor_stride == 4
    assert args.matrix_continue_on_model_error is True

    default_args = probe.parse_args(["--probe-model", "matrix-zoo", "--matrix-model-tier", "first_wave"])
    assert "timm_mobilenetv3_large_100_tsm_tcn" in default_args.matrix_model_ids
    assert "hf_videomae_base_kinetics" not in default_args.matrix_model_ids

    with pytest.raises(SystemExit):
        probe.parse_args(["--probe-model", "mobilenetv3", "--matrix-model-ids", "timm_resnet18_tcn"])


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
    assert "TCN_VARIANTS=" in text
    assert "lite dilated multiscale motion residual gated separable_dilated causal_dilated" in text
    assert "ms_tcnpp c2f_tcn asformer_lite fact_lite temporal_mamba_lite" in text
    assert "--mobilenet-sizes" not in text
    assert "SLURM_STEP_GPUS" in text


def test_matrix_zoo_gpu1_launchers_fail_close_and_use_matrix_probe():
    image_text = MATRIX_IMAGE_GPU1_SCRIPT.read_text(encoding="utf-8")
    video_text = MATRIX_VIDEO_GPU1_SCRIPT.read_text(encoding="utf-8")

    for text in (image_text, video_text):
        assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
        assert "--probe-model matrix-zoo" in text
        assert "--matrix-model-ids ${MODEL_IDS}" in text
        assert "--matrix-continue-on-model-error" in text
        assert "--save-checkpoint" in text
        assert "CUDA_VISIBLE_DEVICES=0" not in text
    assert "timm_convnext_tiny_tcn" in image_text
    assert "torchvision_r3d_18" in video_text
    assert "pytorchvideo_x3d_xs" in video_text


def test_matrix_model_directory_layout_is_model_specific():
    probe = load_probe_module()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = probe._probe_out_dir(
            root,
            probe_model="matrix-zoo",
            spatial_size=64,
            multi_size=False,
            tcn_variant="hf/name:with-colon",
            multi_variant=True,
        )

        assert path == root / "matrix_zoo_hf_name_with-colon_64"


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


def test_build_probe_model_supports_matrix_zoo_branch(monkeypatch):
    probe = load_probe_module()
    captured = {}

    class FakeMatrixZoo:
        def __init__(
            self,
            *,
            model_id,
            pretrained,
            freeze_backbone,
            temporal_hidden_dim,
            video_clip_len,
            video_anchor_stride,
        ):
            captured.update(
                {
                    "model_id": model_id,
                    "pretrained": pretrained,
                    "freeze_backbone": freeze_backbone,
                    "temporal_hidden_dim": temporal_hidden_dim,
                    "video_clip_len": video_clip_len,
                    "video_anchor_stride": video_anchor_stride,
                }
            )

    monkeypatch.setattr(probe, "C3MatrixZooActionProbe", FakeMatrixZoo)
    args = probe.parse_args(
        [
            "--probe-model",
            "matrix-zoo",
            "--matrix-model-ids",
            "timm_resnet18_tcn",
            "--no-matrix-pretrained",
            "--no-matrix-freeze-backbone",
            "--matrix-temporal-hidden-dim",
            "64",
            "--matrix-video-clip-len",
            "8",
            "--matrix-video-anchor-stride",
            "4",
        ]
    )
    args.matrix_model_id = "timm_resnet18_tcn"

    model, reader_cfg = probe._build_probe_model(args, cfg=types.SimpleNamespace(), spatial_size=64)

    assert isinstance(model, FakeMatrixZoo)
    assert reader_cfg is None
    assert captured == {
        "model_id": "timm_resnet18_tcn",
        "pretrained": False,
        "freeze_backbone": False,
        "temporal_hidden_dim": 64,
        "video_clip_len": 8,
        "video_anchor_stride": 4,
    }


def test_matrix_zoo_video_classifier_replacement_supports_conv3d_head():
    probe = load_probe_module()
    torch = pytest.importorskip("torch")
    nn = torch.nn

    scout = object.__new__(probe.C3MatrixZooActionProbe)
    scout.model_id = "fake_s3d"
    scout.backbone = types.SimpleNamespace(
        classifier=nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv3d(1024, 400, kernel_size=1),
        )
    )

    scout._replace_video_classifier(nn)

    assert isinstance(scout.backbone.classifier[-1], nn.Conv3d)
    assert scout.backbone.classifier[-1].in_channels == 1024
    assert scout.backbone.classifier[-1].out_channels == 1


def test_temporal_tcn_new_variants_keep_framewise_shape_and_mask_invalid_positions(monkeypatch):
    probe = load_probe_module()

    class FakeScalar:
        def __gt__(self, other):
            return self

        def item(self):
            return False

    class FakeTensor:
        def __init__(self, shape, *, ndim=None, device="cpu"):
            self.shape = tuple(shape)
            self.ndim = len(self.shape) if ndim is None else ndim
            self.device = device
            self.masked_fill_value = None

        def float(self):
            return self

        def detach(self):
            return self

        def abs(self):
            return self

        def amax(self):
            return FakeScalar()

        def reshape(self, *shape):
            resolved = []
            known = 1
            unknown_idx = None
            total = 1
            for dim in self.shape:
                total *= int(dim)
            for idx, dim in enumerate(shape):
                if int(dim) == -1:
                    unknown_idx = idx
                    resolved.append(1)
                else:
                    resolved.append(int(dim))
                    known *= int(dim)
            if unknown_idx is not None:
                resolved[unknown_idx] = total // known
            return FakeTensor(tuple(resolved), device=self.device)

        def flatten(self, start_dim=0):
            if int(start_dim) != 1:
                raise AssertionError("test fake only supports flatten(1)")
            tail = 1
            for dim in self.shape[1:]:
                tail *= int(dim)
            return FakeTensor((self.shape[0], tail), device=self.device)

        def transpose(self, dim0, dim1):
            shape = list(self.shape)
            shape[int(dim0)], shape[int(dim1)] = shape[int(dim1)], shape[int(dim0)]
            return FakeTensor(tuple(shape), device=self.device)

        def squeeze(self, dim):
            shape = list(self.shape)
            if shape[int(dim)] == 1:
                shape.pop(int(dim))
            return FakeTensor(tuple(shape), device=self.device)

        def to(self, *args, **kwargs):
            return self

        def bool(self):
            return self

        def __invert__(self):
            return FakeTensor(self.shape, device=self.device)

        def masked_fill(self, mask, value):
            result = FakeTensor(self.shape, device=self.device)
            result.masked_fill_value = value
            result.masked_fill_mask_shape = mask.shape
            return result

        def __add__(self, other):
            return FakeTensor(self.shape, device=self.device)

    class FakeModule:
        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            raise TypeError(f"{self.__class__.__name__} has no forward")

        def train(self):
            return self

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def parameters(self):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return state_dict

    class FakeSequential(FakeModule):
        def __init__(self, *modules):
            self.modules = modules

        def __call__(self, x):
            for module in self.modules:
                x = module(x)
            return x

    class FakeModuleList(list):
        pass

    class FakeConv(FakeModule):
        def __init__(self, in_channels, out_channels, *args, **kwargs):
            self.in_channels = in_channels
            self.out_channels = out_channels

        def __call__(self, x):
            shape = list(x.shape)
            shape[1] = int(self.out_channels)
            return FakeTensor(tuple(shape), device=x.device)

    class FakeIdentityModule(FakeModule):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class FakeAdaptiveAvgPool2d(FakeModule):
        def __init__(self, output_size):
            self.output_size = output_size

        def __call__(self, x):
            return FakeTensor((x.shape[0], x.shape[1], 1, 1), device=x.device)

    class FakeGLU(FakeModule):
        def __init__(self, dim=1):
            self.dim = int(dim)

        def __call__(self, x):
            shape = list(x.shape)
            shape[self.dim] = shape[self.dim] // 2
            return FakeTensor(tuple(shape), device=x.device)

    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = FakeModule
    fake_nn.Sequential = FakeSequential
    fake_nn.ModuleList = FakeModuleList
    fake_nn.Conv2d = FakeConv
    fake_nn.Conv1d = FakeConv
    fake_nn.BatchNorm2d = FakeIdentityModule
    fake_nn.BatchNorm1d = FakeIdentityModule
    fake_nn.SiLU = FakeIdentityModule
    fake_nn.Dropout = FakeIdentityModule
    fake_nn.AdaptiveAvgPool2d = FakeAdaptiveAvgPool2d
    fake_nn.GLU = FakeGLU
    fake_torch = types.SimpleNamespace(nn=fake_nn)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setattr(probe, "_import_torch", lambda: (fake_torch, None))

    frames = FakeTensor((2, 5, 3, 16, 16))
    valid = FakeTensor((2, 5), ndim=2)

    for variant in ("residual", "gated", "separable_dilated", "causal_dilated"):
        model = probe.C3TemporalTCNActionProbe(variant=variant, spatial_size=16, hidden_dim=32)
        logits = model(frames, valid)

        assert logits.shape == (2, 5)
        assert logits.masked_fill_value == 0.0
        assert logits.masked_fill_mask_shape == (2, 5)


def test_temporal_segmentation_reader_variants_keep_real_torch_framewise_shape():
    probe = load_probe_module()
    torch = pytest.importorskip("torch")

    frames = torch.rand(2, 7, 3, 16, 16)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)

    for variant in ("ms_tcnpp", "c2f_tcn", "asformer_lite", "fact_lite", "temporal_mamba_lite"):
        model = probe.C3TemporalTCNActionProbe(variant=variant, spatial_size=16, hidden_dim=32, dropout=0.0)
        model.eval()
        with torch.no_grad():
            logits = model(frames, valid)

        assert logits.shape == (2, 7)
        assert torch.all(logits[~valid] == 0)
        assert torch.isfinite(logits[valid]).all()


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
            "final_val": {
                "average_precision": 0.55,
                "roc_auc": 0.61,
                "indirect_selection_quality": {
                    "strategy_comparison": {
                        "best_boundary_support_strategy": "delta_p_action",
                        "boundary_support_r1_by_strategy": {"delta_p_action": 0.70},
                    }
                },
            },
            "out_dir": "root/temporal_tcn_lite_64",
        },
        {
            "probe_model": "temporal-tcn",
            "tcn_variant": "motion",
            "spatial_size": 64,
            "final_val": {
                "average_precision": 0.66,
                "roc_auc": 0.72,
                "indirect_selection_quality": {
                    "strategy_comparison": {
                        "best_boundary_support_strategy": "weighted_transition_mix",
                        "boundary_support_r1_by_strategy": {"weighted_transition_mix": 0.80},
                    }
                },
            },
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
    assert combined["comparison"]["best_indirect_strategy_by_variant"]["lite"] == "delta_p_action"
    assert combined["comparison"]["best_indirect_strategy_by_variant"]["motion"] == "weighted_transition_mix"


def test_matrix_model_summary_exposes_per_model_results_and_failures():
    probe = load_probe_module()

    summaries = [
        {
            "probe_model": "matrix-zoo",
            "matrix_model_id": "timm_resnet18_tcn",
            "spatial_size": 64,
            "final_val": {
                "average_precision": 0.60,
                "roc_auc": 0.70,
                "best_f1": 0.55,
                "indirect_selection_quality": {
                    "strategy_comparison": {
                        "best_boundary_support_strategy": "delta_p_action",
                        "boundary_support_r1_by_strategy": {"delta_p_action": 0.75},
                    }
                },
            },
            "out_dir": "root/matrix_zoo_timm_resnet18_tcn_64",
        },
        {
            "probe_model": "matrix-zoo",
            "matrix_model_id": "hf_videomae_small_kinetics",
            "status": "failed",
            "error": {"type": "ValueError", "message": "transformers adapter not enabled"},
            "out_dir": "root/matrix_zoo_hf_videomae_small_kinetics_64",
        },
    ]

    combined = probe._combine_matrix_model_summaries(
        base_summary={"probe_model": "matrix-zoo", "seed": 3},
        summaries=summaries,
        args_out_dir=Path("root"),
    )

    assert combined["schema_version"] == "lowres_action_probe_matrix_zoo_v1"
    assert combined["matrix_model_ids"] == ["timm_resnet18_tcn", "hf_videomae_small_kinetics"]
    assert combined["comparison"]["best_average_precision_model"] == "timm_resnet18_tcn"
    assert combined["comparison"]["best_indirect_strategy_by_model"]["timm_resnet18_tcn"] == "delta_p_action"
    assert "hf_videomae_small_kinetics" in combined["comparison"]["failed_models"]


def test_load_torch_state_dict_accepts_probe_state_dict(tmp_path):
    probe = load_probe_module()
    import torch

    path = tmp_path / "probe_reader.pth"
    torch.save({"probe_state_dict": {"backbone.weight": torch.tensor([1.0])}}, path)

    loaded = probe._load_torch_state_dict(str(path))
    assert "backbone.weight" in loaded
    assert torch.equal(loaded["backbone.weight"], torch.tensor([1.0]))
