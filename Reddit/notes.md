## Data
All data is present in `data/`. Train/val/test splits can be regenerated using `python3 train-test-split.py`.

## Model
To train the model from scratch (in our case, pretrained BERT without finetuning), run:
1. `python3 -W ignore domain-tuning.py --data_dir="data/train-test-split/" --bert_model="bert-base-uncased" --output_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=64 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`
2. `python3 -W ignore task-tuning.py --data_dir="data/train-test-split/" --bert_model="bert-base-uncased" --output_dir="trained_model/" --trained_model_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=64 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`

To see performance of the model after task-tuning, add `--do_eval` or `--do_test` to the task-tuning script.

## Evaluation
To see the performance of the model at any time, run `python3 -W ignore test.py --data_dir="data/train-test-split/" --bert_model="bert-base-uncased" --output_dir="error_analysis_output/" --trained_model_dir="trained_model/" --max_seq_length=128 --do_test --eval_batch_size=1 --seed=2019`
