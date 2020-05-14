#!/bin/bash
python hack_train.py --name $1 --data "/content/data" --gpu --batch-size 256 --epochs 12 --learning-rate 0.01 --factor 0.5 --loss "wing_loss"
python hack_train.py --name $2 --data "/content/data" --gpu --batch-size 256 --epochs 12 --learning-rate 0.005 --previous_name $1 --factor 0.5 --best-val-loss 4.1442108 --loss "wing_loss"
python hack_train.py --name $3 --data "/content/data" --gpu --batch-size 256 --epochs 12 --learning-rate 0.00135 --previous_name $2 --factor 0.5 --patience 2 --best-val-loss 3.714837 --loss "wing_loss"