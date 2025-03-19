import os

from smearly.tools.image_files import generate_new_img_dir
from smearly.ml_logic.preprocessing import create_image_dataset
from smearly.ml_logic.model import initialize_enb0_model_layers, compile_model, train_model

from smearly.ml_logic.registry import save_model

def preprocess() -> None:

    if os.environ.get("WITH_GPU"):
        reduced_ds_spec = {
            'train': {
                'healthy': 18992,
                'rubbish': 27132,
                'unhealthy_bothcells': 4070,
                'unhealthy_bothcells_augmented': 4070
            },
            'val': {
                'healthy': 7907,
                'rubbish': 13721,
                'unhealthy_bothcells': 1628
            }
        }
    else:
        reduced_ds_spec = {
            'train': {
                'healthy': 950,
                'rubbish': 1357,
                'unhealthy_bothcells': 203,
                'unhealthy_bothcells_augmented': 203
            },
            'val': {
                'healthy': 395,
                'rubbish': 686,
                'unhealthy_bothcells': 81
            }
        }

    generate_new_img_dir(reduced_ds_spec, all_img_basedir='./raw_data/all', target_dir='./raw_data/rebalanced')

def train() -> None:
    data = "./raw_data/rebalanced"
    train_ds = create_image_dataset(directory=os.path.join(data, "train"), normalize=False)
    val_ds = create_image_dataset(directory=os.path.join(data, "val"), normalize=False)
    model = initialize_enb0_model_layers((224, 224, 3))
    model = compile_model(model, learning_rate=0.001)
    model, history = train_model(model = model,
                                 train_data = train_ds,
                                 validation_data = val_ds,
                                 batch_size=32,
                                 epochs=100,
                                 fine_tuning=True)
    save_model(model)
    print("🎉 Model training finished")
