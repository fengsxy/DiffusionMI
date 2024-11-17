#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate mutual_infomation


python benchmark.py  --task_start 0 --task_end 5&
python benchmark.py  --task_start 5 --task_end 10&
python benchmark.py  --task_start 10 --task_end 15&
python benchmark.py  --task_start 15 --task_end 20&