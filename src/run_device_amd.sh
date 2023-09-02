
python main.py \
  --task device-amd \
  --num_epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --warmup_steps 0.25 \
  --random_seed 42 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
