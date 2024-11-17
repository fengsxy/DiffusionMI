#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate mutual_infomation


python self_consistency_ae_benchmark.py  --row_start 0 --row_end 2&
python self_consistency_ae_benchmark.py  --row_start 2 --row_end 4&
python self_consistency_ae_benchmark.py  --row_start 4 --row_end 6&
python self_consistency_ae_benchmark.py  --row_start 6 --row_end 8&