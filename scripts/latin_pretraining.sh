python pretraining.py \
	--output_dir logs/latin_pretraining_w19cls -c config/latin_19class.py --dataset_file  synthetic_latin_diagram \
	--options dn_scalar=100 embed_init_tgt=TRUE