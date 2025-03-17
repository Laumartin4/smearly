from smearly.tools.image_files import generate_new_img_dir

def preprocess() -> None:

    reduced_ds_spec = {
        'train': {
            'bothcells': 0,
            'healthy': 950,
            'rubbish': 1357,
            'unhealthy': 0,
            'unhealthy_bothcells': 203,
            'unhealthy_bothcells_augmented': 203
        },
        'val': {
            'bothcells': 0,
            'healthy': 395,
            'rubbish': 686,
            'unhealthy': 0,
            'unhealthy_bothcells': 81,
            'unhealthy_bothcells_augmented': 0
        }
    }

    generate_new_img_dir(reduced_ds_spec, all_img_basedir='./raw_data/all', target_dir='./raw_data/reduced')

def train() -> None:
    pass
