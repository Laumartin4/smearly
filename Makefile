.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y smearly || :
	@pip install -e .

download_train_images:
	mkdir -p raw_data/all
	wget -P raw_data/all/ --content-disposition 'https://drive.usercontent.google.com/download?id=1rhrLuk-Gt7BN8G6fgJtWn1pQrPfiu_9e&export=download&authuser=0&confirm=t&uuid=4e1ecacb-7813-4231-a700-78aeb639ce2f&at=AEz70l6GNCaI-nPF63PrhmOF3j0P%3A1741964868702'
	7z x -o raw_data/all/ isbi2025-ps3c-test-dataset.7z

merge_unhealthy_bothcells:
	mkdir -p raw_data/all/unhealthy_bothcells
	[ -d unhealthy_bothcells ] || cp -a raw_data/all/unhealthy/* raw_data/all/bothcells/* raw_data/all/unhealthy_bothcells

download_augmented_images:
	if [ \! -d raw_data/all/unhealthy_augmented ] ; then gsutil -m cp -r "gs://smearly-data/unhealthy augmented" raw_data/all/ ; mv 'raw_data/all/unhealthy augmented' 'raw_data/all/unhealthy_augmented' ; fi

download_all_images:
	$(MAKE) download_train_images
	$(MAKE) merge_unhealthy_bothcells
	$(MAKE) download_augmented_images

run_preprocess:
	python -c 'from smearly.interface.main import preprocess; preprocess()'

run_train:
	#python smearly/ml_logic/model.py
	python -c 'from smearly.interface.main import train; train()'

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
