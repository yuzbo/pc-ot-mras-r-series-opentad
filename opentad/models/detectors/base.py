import torch
from ..utils.post_processing import load_predictions, save_predictions


class BaseDetector(torch.nn.Module):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        if getattr(self, "token_compressor", None) is not None:
            if getattr(infer_cfg, "load_from_raw_predictions", False) or getattr(infer_cfg, "save_raw_prediction", False):
                raise ValueError(
                    "token_compressor forbids raw-prediction load/save because cached predictions bypass "
                    "compressed-axis metadata updates"
                )
        if self._uses_pc_ot_mras_live_reader_path():
            if getattr(infer_cfg, "load_from_raw_predictions", False) or getattr(infer_cfg, "save_raw_prediction", False):
                raise ValueError(
                    "PC-OT-MRAS forbids raw-prediction load/save because cached predictions bypass "
                    "the live ActionFormer reader/bridge path"
                )

        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results

    def _uses_pc_ot_mras_live_reader_path(self):
        if getattr(self, "pc_ot_mras_reader", None) is not None:
            return True
        neck = getattr(self, "neck", None)
        if neck is None:
            return False
        return neck.__class__.__name__ == "PCOTMRASDetectorBridge"
