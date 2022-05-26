#!/bin/bash

for $NUM_MODEL in {0..19};do
	mkdir -p checkpoints/deepcluster/$NUM_MODEL
	python -m torch.distributed.launch --nproc_per_node=1 ./main_sign2vec.py \
			--arch resnet18 \
			--data_path ./data/cyprominoan/dataset \
			--dist_url env:// \
			--epochs 100 \
			--batch_size 16 \
			--size_crops 80 60 \
			--nmb_crops 6 10 \
			--min_scale_crops 0.6 0.4 \
			--max_scale_crops 1.0 0.6 \
			--crops_for_assign 0 \
			--freeze_prototypes_niters 300000 \
			--wd 0.000001 \
			--warmup_epochs 10 \
			--start_warmup 0.3 \
			--temperature 0.1 \
			--feat_dim 128 \
			--nmb_prototypes 100 100 100 \
			--base_lr 4.8 \
			--final_lr 0.0048 \
			--lambda_cbow 0.0 \
			--dump_path "checkpoints/deepcluster/$NUM_MODEL"
	done
