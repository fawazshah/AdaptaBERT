#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=fs2217 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/fs2217/miniconda3/bin/:$PATH
source activate
source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
TERM=vt100 # or TERM=xterm

python3 -W ignore domain-tuning.py --data_dir="data/train-test-split/" --bert_model="bert-base-uncased" --output_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=64 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16
echo Done!
