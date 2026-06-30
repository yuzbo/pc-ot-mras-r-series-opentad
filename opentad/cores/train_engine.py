import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def _parse_nonfinite_loss_guard(nonfinite_loss_guard):
    if nonfinite_loss_guard is None or nonfinite_loss_guard is False:
        return dict(enabled=False, max_skips=0, max_consecutive_skips=0)
    if nonfinite_loss_guard is True:
        return dict(enabled=True, max_skips=0, max_consecutive_skips=0)
    if not isinstance(nonfinite_loss_guard, dict):
        raise ValueError("nonfinite_loss_guard must be None, bool, or dict")
    return dict(
        enabled=bool(nonfinite_loss_guard.get("enabled", False)),
        max_skips=int(nonfinite_loss_guard.get("max_skips", 0)),
        max_consecutive_skips=int(nonfinite_loss_guard.get("max_consecutive_skips", 0)),
    )


def _nonfinite_loss_names(losses):
    names = []
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        if not torch.isfinite(value.detach()).all():
            names.append(key)
    return names


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    max_train_iters=None,
    nonfinite_loss_guard=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True
    guard = _parse_nonfinite_loss_guard(nonfinite_loss_guard)
    nonfinite_skip_count = 0
    consecutive_nonfinite_skip_count = 0

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        if max_train_iters is not None and iter_idx >= max_train_iters:
            logger.info(f"[Train]: max_train_iters={max_train_iters} reached, stop epoch early")
            break
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        nonfinite_names = _nonfinite_loss_names(losses)
        if guard["enabled"] and nonfinite_names:
            nonfinite_skip_count += 1
            consecutive_nonfinite_skip_count += 1
            optimizer.zero_grad()
            logger.warning(
                "[Train]: non-finite loss detected at epoch {:d} iter {:d}; skip optimizer.step; names={}".format(
                    curr_epoch,
                    iter_idx,
                    nonfinite_names,
                )
            )
            if (
                (guard["max_skips"] >= 0 and nonfinite_skip_count > guard["max_skips"])
                or (
                    guard["max_consecutive_skips"] >= 0
                    and consecutive_nonfinite_skip_count > guard["max_consecutive_skips"]
                )
            ):
                raise RuntimeError(
                    "Non-finite training loss exceeded guard threshold: "
                    f"total_skips={nonfinite_skip_count}, "
                    f"consecutive_skips={consecutive_nonfinite_skip_count}, "
                    f"names={nonfinite_names}"
                )
            continue
        consecutive_nonfinite_skip_count = 0

        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
