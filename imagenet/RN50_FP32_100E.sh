# Train on 1 Node with 1 GPU 
python imagenet_train.py  -a fixup_resnet50 --resume /home/hj14/Bias-Free-Fixup/imagenet/checkpoint-16.pth.tar --epochs 100 -b 64 --gpu 0  [path to imagenet]
# Train on 1 Node with N GPU
python imagenet_train.py  -a fixup_resnet50 --resume /home/hj14/Bias-Free-Fixup/imagenet/checkpoint-16.pth.tar --epochs 100 -b 64 [path to imagenet]
# Train on N Node with M GPU each
python imagenet_train.py  -a fixup_resnet50 --rank 0 --dist-url tcp://127.0.0.1:23456 --world-size N --epochs 100 -b 64  --multiprocessing-distributed [path to imagenet]
python imagenet_train.py  -a fixup_resnet50 --rank 1 --dist-url tcp://127.0.0.1:23456 --world-size N --epochs 100 -b 64  --multiprocessing-distributed [path to imagenet]