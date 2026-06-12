python finetuning.py --config_file config/finetuningk4.py --output_dir logs/finetuning_k4 \
 --dataset_file eida --pretrain_model_path ./logs/pretraining_k4/checkpoint0039.pth \
 --options dn_scalar=100 embed_init_tgt=TRUE