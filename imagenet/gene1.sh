#!/bin/bash

for i in {0..90}
do
  j=$[i*1251]
  echo $i
  echo $j
  echo "python ckpt_sparsity.py --checkpoint=/scratch/hj14/tpu_checkpoints/ckpt-2/model.ckpt-$j --sparsity_technique=magnitude_pruning --model=rn50 | tee -a sparsity-ckpt.csv " >> sparsity_script.sh
done 
