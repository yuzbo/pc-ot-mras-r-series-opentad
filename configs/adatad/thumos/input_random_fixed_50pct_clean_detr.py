_base_ = ["./input_random_fixed_50pct_tadtr_sparse_pilot.py"]

model = dict(
    projection=dict(
        _delete_=True,
        type="DensePassthroughConv1DTransformerProj",
        in_channels=384,
        out_channels=256,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=384,
        input_pdrop=0.2,
    ),
    neck=None,
    transformer=dict(
        encoder=dict(num_feature_levels=1),
        decoder=dict(num_feature_levels=1),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_clean_detr"