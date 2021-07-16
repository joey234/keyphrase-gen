/usr/bin/python3.7 main.py --model_dir t5_base \
                  --data_dir data \
                  --seed 1 \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 1 \
                  --tuning_metric f1 \
                  --gpu_id 0 \
                  --learning_rate 5e-5 \
                  --train_batch_size 2 \
                  --eval_batch_size 2 