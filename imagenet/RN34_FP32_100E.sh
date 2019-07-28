# batch size option: -b, I keep the default value for other options like learning rate, momentum, weight decay in the repo 
# Train on 1 Node with 1 GPU 
python imagenet_train.py  -a fixup_resnet34 --resume --epochs 100 -b 64 --gpu 0  [path to imagenet]
# Train on 1 Node with N GPU
python imagenet_train.py  -a fixup_resnet34 --resume --epochs 100 -b 64 [path to imagenet]
# Train on N Node with M GPU each
python imagenet_train.py  -a fixup_resnet34 --rank 0 --dist-url tcp://127.0.0.1:23456 --world-size N --epochs 100 -b 64  --multiprocessing-distributed [path to imagenet]
python imagenet_train.py  -a fixup_resnet34 --rank 1 --dist-url tcp://127.0.0.1:23456 --world-size N --epochs 100 -b 64  --multiprocessing-distributed [path to imagenet]
