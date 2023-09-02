
python main.py \
  --task thread-tahiti \
  --num_epochs 20 \
  --batch_size 8 \
  --learning_rate 6e-5 \
  --warmup_steps 0.13 \
  --random_seed 23 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
