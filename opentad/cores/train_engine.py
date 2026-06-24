import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def _assert_loss_dict_finite(losses, *, stage):
    if not isinstance(losses, dict):
        raise TypeError(f"{stage} must return a loss dict, got {type(losses)!r}")
    if "cost" not in losses:
        raise KeyError(f"{stage} loss dict must contain cost")
    for key, value in losses.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        finite = torch.isfinite(value.detach()).all()
        if not bool(finite.item()):
            raise FloatingPointError(f"{stage} produced non-finite loss {key}")
    cost = losses["cost"]
    if not torch.is_tensor(cost):
        cost = torch.as_tensor(cost)
    if not bool(torch.isfinite(cost.detach()).all().item()):
        raise FloatingPointError(f"{stage} produced non-finite cost")


def _format_nonfinite_grad(name, grad):
    finite = torch.isfinite(grad)
    finite_values = grad[finite]
    max_abs = finite_values.float().abs().max().item() if finite_values.numel() else None
    nan_count = torch.isnan(grad).sum().item()
    inf_count = torch.isinf(grad).sum().item()
    return (
        f"{name}: shape={tuple(grad.shape)} "
        f"nan={int(nan_count)} inf={int(inf_count)} max_abs_finite={max_abs}"
    )


def _assert_grad_norm_finite(model):
    total_sq = None
    named_parameters = model.named_parameters() if hasattr(model, "named_parameters") else []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all().item()):
            detail = _format_nonfinite_grad(name, grad)
            raise FloatingPointError(f"training produced non-finite parameter gradient: {detail}")
        grad_norm = grad.float().norm(2)
        total_sq = grad_norm.pow(2) if total_sq is None else total_sq + grad_norm.pow(2)
    if total_sq is None:
        return None
    total_norm = total_sq.sqrt()
    if not bool(torch.isfinite(total_norm).all().item()):
        raise FloatingPointError("training produced non-finite gradient norm")
    return total_norm


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
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    if max_train_iters is not None:
        max_train_iters = int(max_train_iters)
        if max_train_iters <= 0:
            raise ValueError("max_train_iters must be positive when provided")
        num_iters = min(num_iters, max_train_iters)
    use_amp = False if scaler is None else True

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
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
        _assert_loss_dict_finite(losses, stage="train forward")

        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"]).backward()
            scaler.unscale_(optimizer)
        else:
            losses["cost"].backward()
        _assert_grad_norm_finite(model)

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
            if not bool(torch.isfinite(grad_norm.detach()).all().item()):
                raise FloatingPointError("training produced non-finite clipped gradient norm")

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
        if max_train_iters is not None and (iter_idx + 1) >= max_train_iters:
            logger.info("[Train]: max_train_iters=%d reached; ending smoke epoch early", max_train_iters)
            break


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
