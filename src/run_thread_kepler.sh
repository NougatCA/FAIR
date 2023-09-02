
python main.py \
  --task thread-kepler \
  --num_epochs 20 \
  --batch_size 8 \
  --learning_rate 8e-5 \
  --warmup_steps 0.137 \
  --random_seed 666 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
