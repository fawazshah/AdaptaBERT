#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=fs2217 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/fs2217/miniconda3/bin/:$PATH
source activate
source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
TERM=vt100 # or TERM=xterm

echo "#################"
echo "MLM DOMAIN TUNING"
echo "#################"

python3 -W ignore mlm-domain-tuning.py --data_dir="data/" --bert_model="bert-base-cased" --output_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=5 --warmup_proportion=0.1 --seed=2019 --fp16

echo "#################"
echo "NSP DOMAIN TUNING"
echo "#################"

python3 -W ignore nsp-domain-tuning.py --data_dir="data/" --bert_model="bert-base-cased"  --trained_model_dir="lm_output" --output_dir="lm_nsp_output" --max_seq_length=128 --do_train --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=1 --warmup_proportion=0.1 --seed=2019 --fp16

echo "###########"
echo "TASK TUNING"
echo "###########"

python3 -W ignore task-tuning.py --data_dir="data/" --bert_model="bert-base-cased" --trained_model_dir="lm_nsp_output/" --output_dir="trained_model/" --max_seq_length=128 --do_train --do_eval --do_test --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=5 --warmup_proportion=0.1 --seed=2019 --fp16

echo Done!
