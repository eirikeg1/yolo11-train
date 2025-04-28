#!/bin/bash

python train_yolo.py \
  --out_path /home/eirik/Projects/yolo11-train/data \
  --data_dir /home/eirik/Projects/data/SoccerNetGS \
  --finetune_class all \
  --iou 0.45 \
  --imagesz 640,1280 \
  --epochs 2


