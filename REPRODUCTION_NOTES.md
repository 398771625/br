# FusionVLAD NCLT reproduction notes

This is an independent clean reimplementation. It does not import the BEVPlace++, REIN, Range, BEV–Range fusion, Scan Context descriptor, or CVTNet network. The Scan Context NPZ files are used **only** as immutable frame-order manifests. CVTNet's `get_velo()` is reused only as the verified raw NCLT binary decoder.

## 1. Confirmed from the FusionVLAD paper

- Dual top-down and spherical views with independent branches and VLAD aggregation.
- Image size `K=64`, spherical slices `M=5`, maximum spherical range 30 m, and five 6 m range slices.
- Top-down channels are maximum height and mean LiDAR intensity.
- Spherical channels are nearest range and ray/surface-normal angle.
- Metric tuples use positives below 5 m and negatives above 10 m.
- Yaw augmentation uses 30-degree intervals.
- Adam optimization with initial learning rate `1e-4`.

## 2. Confirmed by the later CVTNet FusionVLAD reimplementation

- The final fused FusionVLAD descriptor is 2048-D. This implementation realizes it as `[f_top(512), f_joint(1024), f_spherical(512)]`, followed by L2 normalization.

## 3. Reimplementation choices in this project

> Because NCLT sequence 2012-01-08 was unavailable in the local dataset,
> FusionVLAD was trained only on the reference traversal 2012-01-15.
> All six cross-date query traversals were strictly held out from training,
> validation, normalization, and checkpoint selection.

This is **not** claimed to be the original FusionVLAD paper's training protocol.

- Fixed frame-order 80/20 source split: first 80% training, last 20% source validation; seed 1024.
- Deterministic temporal local map with `local_window=2`, rather than an incompletely specified octree-uncertainty implementation.
- Voxel size 0.20 m, XY crop `[-32,32)`, and boundary frames using only available neighbors.
- Surface-normal radius 1.5 m and maximum 30 neighbors.
- 64 NetVLAD clusters because the original paper did not clearly report the cluster count.
- VGG16-style branches. Optional ImageNet initialization and first-convolution channel adaptation by repeating mean RGB weights are reimplementation choices.
- Fully connected parallel fusion: two 512-D branches, 1024-D joint feature, and 2048-D concatenated output.
- 30 epochs and batch of 2 queries (batch 1 is permitted for memory limits and recorded in `config_used.yaml`).
- PointNetVLAD-inspired/lazy reimplementation choices: translation margin 0.5, viewpoint/rotation margin 0.2, 2 positives, and 18 negatives per query. The loss takes the hardest positive and hardest negative and applies both translation-ranking and viewpoint-positive penalties independently to top, spherical, and final descriptors.
- Cached top-down image rotation uses bilinear sampling; spherical azimuth rotation uses periodic roll with linear interpolation. Both views receive the same sampled yaw.
- The best checkpoint is selected only by source-validation Recall@1; no cross-date query is read during training or selection.
