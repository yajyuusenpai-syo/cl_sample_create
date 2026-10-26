# Upstream attribution

The Spatial-CLAP architecture and vendored reference sources in `upstream/` come
from https://github.com/sarulab-speech/SpatialCLAP at commit
`4ef788b8a6d198263420c1e6b15623591f443dd4`.

The compact `offline_teacher/weights/spatial_encoder.pt` contains the unchanged
spatial submodule tensors extracted from the authors' published checkpoint:
https://huggingface.co/sarulab-speech/SpatialCLAP/resolve/main/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt

Provenance and hashes are recorded in `offline_teacher/teacher_provenance.json`.
Original authorship notices in reference sources are retained. Refer to the
upstream project/model for their applicable terms; this repository does not
grant new rights to third-party code, model weights, or datasets.

The smoke-test audio in this repository is generated from deterministic
synthetic sine mixtures. No source dataset audio/video or computed dataset
features are included.

PyTorch, CUDA libraries, FFmpeg, NumPy, SciPy and the other wheel dependencies
retain their respective licenses. The large wheel binaries are not vendored;
`pkgs/manifest.json` records their exact source URLs and hashes.
