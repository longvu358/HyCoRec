#!/usr/bin/env bash
# Create the `hycorec` conda env for servers with old GPUs (V100 sm_70 / P100 sm_60).
set -euo pipefail
cd "$(dirname "$0")"

conda env create -f environment.yml || conda env update -f environment.yml --prune

conda run -n hycorec python -c "
import torch
print(torch.__version__, torch.version.cuda)
print('arch_list:', torch.cuda.get_arch_list())
print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
x = torch.ones(1, device='cuda:0') * 2
print('kernel test ok:', x.item())
"
echo "Done. Activate with: conda activate hycorec"
