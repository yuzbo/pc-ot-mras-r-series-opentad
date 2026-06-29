_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_scout_spatial_size = 64

model = dict(
    frame_selector=dict(
        scout_spatial_size=c3_scout_spatial_size,
        density_repulsion_loss_weight=0.0,
        scout=dict(
            in_channels=3 * c3_scout_spatial_size * c3_scout_spatial_size,
        ),
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_64px_short_smoke"
