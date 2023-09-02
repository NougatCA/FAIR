
python main.py \
  --task thread-cypress \
  --num_epochs 20 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --warmup_steps 0.13 \
  --random_seed 520 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
