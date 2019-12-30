#!/bin/bash
for i in {1..100..2}
do
  j=$[i+1]
  echo "python3 /home/hj14/ExAct-NoBiasFixup/imagenet/imagenet_train.py  -a fixup_resnet50 --resume /home/hj14/Bias-Free-Fixup/imagenet/resnet50_ckpt/checkpoint-$i.pth.tar --epochs $j  -b 64 --gpu 0  /home/hj14/Imagenet | tee -a sparsity-20.log" >> script.sh
done 
