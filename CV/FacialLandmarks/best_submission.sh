#!/bin/bash
python hack_train.py --name $1 --data "/content/data" --gpu --batch-size 300 --epochs 15 --learning-rate 0.01 --factor 0.5 --loss "huber_loss"
python hack_train.py --name $2 --data "/content/data" --gpu --batch-size 300 --epochs 8 --learning-rate 0.0025 --previous_name $1 --factor 0.2 --best-val-loss 0.484359 --loss "huber_loss"
python hack_train.py --name $3 --data "/content/data" --gpu --batch-size 300 --epochs 7 --learning-rate 0.0001 --previous_name $2 --factor 0.2 --best-val-loss 0.44 --loss "huber_loss"