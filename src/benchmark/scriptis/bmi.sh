#!/bin/bash
#activate the virtual environment
#source ../../venv/bin/activate
#to select algorithm
#select bmi task start end

echo "Running benchmark with smaller dataset..."
nohup python benchmark.py --train_samples 10000 --test_samples 1000 --max_steps 10000 --trials 5&

echo "Running benchmark with larger dataset..."
nohup python benchmark.py --train_samples 100000 --test_samples 10000 --max_steps 100000 --trials 5&

nohup python benchmark.py --train_samples 10000 --test_samples 1000 --max_steps 10000& 

nohup python benchmark.py --train_samples 10000 --test_samples 1000 --max_steps 1000& 

nohup python benchmark.py --train_samples 100000 --test_samples 10000 --max_steps 100000&