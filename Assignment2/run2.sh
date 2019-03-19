#!/bin/bash
#SBATCH --array=1-12
source ~/.bashrc

# 4.3 - Hyperparam tuning
case $SLURM_ARRAY_TASK_ID in
    # RNN
    1)
        python ptb-lm.py --save_dir=4_3_a --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=40 --seq_len=20 --hidden_size=600 --num_layers=2 --dp_keep_prob=0.5
        ;;
    2)
        python ptb-lm.py --save_dir=4_3_b --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=600 --num_layers=2 --dp_keep_prob=0.5
        ;;
    3)
        python ptb-lm.py --save_dir=4_3_c --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=25 --hidden_size=400 --num_layers=2 --dp_keep_prob=0.5
        ;;
    4)
        python ptb-lm.py --save_dir=4_3_d --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=20 --seq_len=35 --hidden_size=800 --num_layers=2 --dp_keep_prob=0.5
        ;;

    # GRU
    5)
        python ptb-lm.py --save_dir=4_3_e --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=25 --hidden_size=1600 --num_layers=2 --dp_keep_prob=0.35 
        ;;
    6)
        python ptb-lm.py --save_dir=4_3_f --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=15 --batch_size=20 --seq_len=35 --hidden_size=1800 --num_layers=2 --dp_keep_prob=0.35
        ;;
    7)
        python ptb-lm.py --save_dir=4_3_g --model=GRU --optimizer=ADAM --initial_lr=0.0003 --batch_size=20 --seq_len=25 --hidden_size=1600 --num_layers=2 --dp_keep_prob=0.35
        ;;
    8)
        python ptb-lm.py --save_dir=4_3_h --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=40 --seq_len=25 --hidden_size=1200 --num_layers=2 --dp_keep_prob=0.35
        ;;

    # Transformer
    9)
        python ptb-lm.py --save_dir=4_3_i --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.00008 --batch_size=80 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
        ;;
    10)
        python ptb-lm.py --save_dir=4_3_j --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0003 --batch_size=170 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
        ;;
    11)
        python ptb-lm.py --save_dir=4_3_k --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=25 --hidden_size=512 --num_layers=8 --dp_keep_prob=0.9
        ;;
    12)
        python ptb-lm.py --save_dir=4_3_l --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.00008 --batch_size=64 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
        ;;

esac

# Run with sbatch --gres=gpu:titanxp:1 -c 1 run2.sh