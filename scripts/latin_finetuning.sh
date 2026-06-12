python finetuning.py \
	--output_dir logs/latin_finetuning_w20cls -c config/latin_20class.py --dataset_file  latin_eida \
	--options dn_scalar=100 embed_init_tgt=TRUE --pretrain_model_path ./eida_checkpoints/latin_pretrain.pth