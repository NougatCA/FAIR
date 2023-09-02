
python main.py \
  --task retrieval-poj104 \
  --num_epochs 20 \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-6 \
  --warmup_steps 0.052 \
  --random_seed 42 \
  --model_path ../models/pre-trained/ \
  --model_size base \
  --single_thread_parsing \
  --use_data_cache \
  --wandb_mode online
