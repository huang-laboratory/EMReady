Place the downloaded EMReady2 model weights in this directory before running inference.

The pretrained model weights can be downloaded from:

- `http://huanglab.phys.hust.edu.cn/EMReady2`

Expected filenames:

- `model_weights/model_0p6.pt`
- `model_weights/model_1p0.pt`
- `model_weights/model_ligand_v0.pt`

The `emready` main command automatically selects `model_0p6.pt` or `model_1p0.pt`
from the input map voxel size when `--model_path` / `-bmp` is not provided.

The `emready ligand` command uses `model_ligand_v0.pt` by default.
