def should_eval_epoch(epoch, workflow):
    """Return True when an eval should run after the zero-based training epoch."""
    val_eval_interval = workflow.val_eval_interval
    if val_eval_interval <= 0:
        return False
    if "val_eval_epochs" not in workflow and "val_eval_interval_anchor_epoch" not in workflow:
        return (epoch + 1) % val_eval_interval == 0

    one_based_epoch = epoch + 1
    explicit_epochs = set(int(item) for item in workflow.get("val_eval_epochs", []))
    if one_based_epoch in explicit_epochs:
        return True
    anchor_epoch = int(workflow.get("val_eval_interval_anchor_epoch", workflow.get("val_start_epoch", 0)))
    if one_based_epoch < anchor_epoch:
        return False
    return (one_based_epoch - anchor_epoch) % val_eval_interval == 0
