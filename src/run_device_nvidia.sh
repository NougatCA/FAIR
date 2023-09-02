
python main.py \
  --task device-nvidia \
  --num_epochs 10 \
  --batch_size 4 \
  --learning_rate 8e-5 \
  --warmup_steps 0.1 \
  --random_seed 4321 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
