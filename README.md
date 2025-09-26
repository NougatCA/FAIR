# FAIR

## Datasets

Datasets can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0g4I8x-zke2YNUK-3ng?e=cpj66P).
Extract the archive file and place the entire folder at the same level as the entire project folder. 
Alternatively, you can place it anywhere else and specify the path using the `--dataset_root` argument.

## Pre-trained Model

Pre-trained model can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0g4I7UoG7qvCRtSMSVQ?e=eyDle3).
Extract the archive file and place the entire folder in the root directory. 
Alternatively, you can place it anywhere else and specify the path using the `--model_path` argument.

## Runs

The basic running script is located in the `src` directory, and you can use 
any additional arguments located in `src/args.py` to change the program's behavior.

## Wandb

Wandb is used to track running; the mode can be set to `offline`(default), `online`, and `disabled`.
If `--wandb_mode` is set to `online`, it requires the user's [wandb](https://wandb.ai) key.

Sign up for an account at [https://wandb.ai](https://wandb.ai) 
and copy your user's key to `wandb_api.key` in the root directory.
