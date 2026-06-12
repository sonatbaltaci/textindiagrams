python finetuning.py --config_file config/finetuning.py --output_dir logs/finetuning \
 --dataset_file eida --pretrain_model_path ./eida_checkpoints/pretrain.pth \
 --options dn_scalar=100 embed_init_tgt=TRUE 
