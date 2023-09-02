
python main.py \
  --task cls-poj104 \
  --num_epochs 5 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --warmup_steps 0.03 \
  --random_seed 42 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
