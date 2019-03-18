#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --array=1-9
source ~/.bashrc

case $SLURM_ARRAY_TASK_ID in
    # 4.1
    1)
        python ptb-lm.py --model=RNN --optimizer=ADAM --save_dir=4_1_a --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
        ;;
    2)
        python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --save_dir=4_1_b --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
        ;;
    3)
        python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --save_dir=4_1_c --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
        ;;

    # 4.2 Note that SGD_LR_SCHEDULE is the default and results can thus be taken from the above
    4)
        python ptb-lm.py --model=RNN --optimizer=SGD --save_dir=4_2_a --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
        ;;
    5)
        python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --save_dir=4_2_b --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
        ;;
    6)
        python ptb-lm.py --model=GRU --optimizer=SGD --save_dir=4_2_c --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
        ;;
    7)
        python ptb-lm.py --model=GRU --optimizer=ADAM --save_dir=4_2_d --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
        ;;
    8)
        python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --save_dir=4_2_e --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
        ;;
    9)
        python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --save_dir=4_2_f --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
        ;;

    # 4.3
    # python ptb-lm.py --model RNN --optimizer SGD

esac

# Run with sbatch --gres=gpu -c 1 run.sh