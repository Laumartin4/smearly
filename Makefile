.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y smearly || :
	@pip install -e .

download_train_images:
	mkdir -p raw_data/all
	wget -P raw_data/all/ --content-disposition 'https://drive.usercontent.google.com/download?id=1q5czzLDE_lQLJlko5Zpb10lUbPwVqkHW&export=download&authuser=0&confirm=t&uuid=f19ce421-e322-4a59-afa8-e7d669fd15e9&at=AEz70l4BbAbUmyZRCll__1846a1T:1742481649161'
	7z x -oraw_data/all/ raw_data/all/isbi2025-ps3c-train-dataset.7z

merge_unhealthy_bothcells:
	mkdir -p raw_data/all/unhealthy_bothcells
	[ -d unhealthy_bothcells ] || cp -a raw_data/all/unhealthy/* raw_data/all/bothcells/* raw_data/all/unhealthy_bothcells

download_augmented_images:
	if [ \! -d raw_data/all/unhealthy_bothcells_augmented ] ; then gsutil -m cp -r "gs://smearly-data/unhealthy augmented" raw_data/all/ ; mv 'raw_data/all/unhealthy augmented' 'raw_data/all/unhealthy_bothcells_augmented' ; fi

download_all_images:
	$(MAKE) download_train_images
	$(MAKE) merge_unhealthy_bothcells
	$(MAKE) download_augmented_images

run_preprocess:
	python -c 'from smearly.interface.main import preprocess; preprocess()'
	mv raw_data/rebalanced/train/unhealthy_bothcells_augmented/* raw_data/rebalanced/train/unhealthy_bothcells/
	rmdir raw_data/rebalanced/train/unhealthy_bothcells_augmented

run_train:
	#python smearly/ml_logic/model.py
	nohup python -c 'from smearly.interface.main import train; train()' > train_$(shell date '+%Y%m%d_%H%m').log &

run_preproc_and_train:
	$(MAKE) run_preprocess
	$(MAKE) run_train

# run_pred:
# 	python -c 'from smearly.interface.main import pred; pred()'

# run_evaluate:
# 	python -c 'from smearly.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate

run_api:
	uvicorn smearly.api.fast:app --reload
