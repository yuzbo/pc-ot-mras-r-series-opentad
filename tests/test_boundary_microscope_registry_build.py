from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABEL = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"


def test_boundary_microscope_lightweight_registry_export_is_visible_without_heavy_deps():
    selectors_init = ROOT / "opentad" / "models" / "selectors" / "__init__.py"
    route_file = ROOT / "opentad" / "models" / "selectors" / "boundary_microscope_acquisition_route.py"

    selectors_text = selectors_init.read_text(encoding="utf-8")
    route_text = route_file.read_text(encoding="utf-8")

    assert "boundary_microscope_acquisition_route" in selectors_text
    assert "BoundaryMicroscopeAcquisitionRoute" in selectors_text
    assert "BOUNDARY_MICROSCOPE_ROUTE_LABEL" in selectors_text
    assert "@SELECTORS.register_module()" in route_text
    assert ROUTE_LABEL in route_text


def test_boundary_microscope_canonical_build_detector_when_runtime_deps_available():
    pytest.importorskip("mmengine", reason="canonical OpenTAD registry build requires mmengine")
    torch = pytest.importorskip("torch", reason="canonical OpenTAD registry build requires torch")
    nn = pytest.importorskip("torch.nn", reason="canonical OpenTAD registry build requires torch.nn")

    try:
        import opentad.models  # noqa: F401
        from opentad.models.builder import MODELS, build_detector
    except Exception as exc:  # pragma: no cover - depends on local mmaction/mmcv install.
        pytest.skip(f"canonical OpenTAD import unavailable, likely missing mmaction/mmcv runtime deps: {exc}")

    class BoundaryMicroscopeRegistryProjection(nn.Module):
        def __init__(self, channels=4, max_seq_len=64):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)
            self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
            nn.init.eye_(self.proj.weight[:, :, 0])

        def forward(self, x, masks):
            return (self.proj(x),), (masks,)

    class BoundaryMicroscopeRegistryHead(nn.Module):
        def __init__(self, in_channels=4):
            super().__init__()
            self.prior_generator = type("_Prior", (), {"strides": [1]})()
            self.score = nn.Conv1d(int(in_channels), 1, kernel_size=1)

        def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None):
            feat = feat_list[0]
            mask = mask_list[0]
            return {"detector_loss": self.score(feat.masked_fill(~mask[:, None, :], 0.0)).square().mean()}

    if "BoundaryMicroscopeRegistryProjection" not in MODELS.module_dict:
        MODELS.register_module()(BoundaryMicroscopeRegistryProjection)
    if "BoundaryMicroscopeRegistryHead" not in MODELS.module_dict:
        MODELS.register_module()(BoundaryMicroscopeRegistryHead)

    model = build_detector(
        dict(
            type="ActionFormer",
            projection=dict(type="BoundaryMicroscopeRegistryProjection", channels=4, max_seq_len=64),
            rpn_head=dict(type="BoundaryMicroscopeRegistryHead", in_channels=4),
            frame_selector=dict(
                type="BoundaryMicroscopeAcquisitionRoute",
                route_label=ROUTE_LABEL,
                meta_key="boundary_microscope_acquisition_plan",
                target_len=32,
                dense_window_size=64,
                max_dense_gap=8,
            ),
        )
    )

    assert model.frame_selector.__class__.__name__ == "BoundaryMicroscopeAcquisitionRoute"
    assert model.frame_selector.route_label == ROUTE_LABEL
