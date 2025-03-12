import os
import sys
import random
import shutil
from tensorflow.image import encode_png as tf_encode_png
from tensorflow.io import write_file as tf_write_file

from smearly.ml_logic.preprocessing import image_file_to_tf, resize_pad_image_tf

def generate_new_img_dir(
    class_nb_files: dict[str, int],
    test_size: float | None = 0.3,
    resize_pad_size: tuple[int,int] | None = (224, 224),
    all_img_basedir: str = '../raw_data/all',
    target_dir: str = '../raw_data/reduced',
    random_seed: int | None = 42
    ) -> None:
    """
    Take some images from subdirs of `all_img_basedir` and put them in `target_dir`.
    Optionally create `target_dir`/(test & train) if test_size is defined.

    Args:
    - class_nb_files: a dictionary with keys being subdir names, and values being
      the number of image files to take, like:
      ```python
          class_nb_files = {
              'bothcells': 30,
              'healthy': 10,
              'unhealthy': 10,
              'rubbish': 10
          }
      ```
    - resize_pad_size (tuple or None): Target size (height, width) of the output
      image (default is 224x224) or None to keep the raw image.
    - all_img_basedir (str): path to the dir containing all images
    - target_dir (str): path to the dir where generated/moved images will be put
    - random_seed (int or None): set to constant to always get the same images
      list with the same class_nb_files

    Returns:
    - image (tf.Tensor): The preprocessed image ready for model input.
    """

    if test_size is not None and not (0 < test_size < 1):
        raise ValueError('test_size must be between 0 and 1')

    subdirs = [f for f in os.listdir(all_img_basedir) if os.path.isdir(os.path.join(all_img_basedir, f))]

    if random_seed is not None:
        random.seed(random_seed)

    files_picked = {}
    for subdir in subdirs:
        if subdir in class_nb_files:
            file_names = [f for f in os.listdir(os.path.join(all_img_basedir, subdir)) if os.path.isfile(os.path.join(all_img_basedir, subdir, f))]
            nb_files_to_pick = class_nb_files[subdir]
            nb_file_names = len(file_names)
            if nb_files_to_pick > nb_file_names:
                print(f'[Warning!] You asked for {nb_files_to_pick} files in {subdir} which contains only {nb_file_names} file(s)', file=sys.stderr)
                print(f'Reducing number of picked files to the maximum: {nb_file_names} file(s)\n', file=sys.stderr)
                nb_files_to_pick = nb_file_names

            files_picked[subdir] = random.sample(file_names, nb_files_to_pick)

    def move_preproc_images(files_picked: dict, target_dir: str):
        for dir, file_names in files_picked.items():
            target_class_dir = os.path.join(target_dir, dir)
            os.makedirs(target_class_dir, exist_ok=True)

            for file_name in file_names:
                source_img_path = os.path.join(all_img_basedir, dir, file_name)
                target_img_path = os.path.join(target_class_dir, file_name)

                if resize_pad_size is not None:
                    preprocessed_image = resize_pad_image_tf(image_file_to_tf(source_img_path), target_size=resize_pad_size, normalize=False)
                    png_image = tf_encode_png(preprocessed_image)
                    tf_write_file(target_img_path, png_image)
                else:
                    shutil.copy2(source_img_path, target_img_path)

    # resample if test_size is specified
    if test_size is not None:
        class_nb_files_test = {k: int(v*0.3) for k, v in class_nb_files.items()}
        #class_nb_files_train = {k: v-class_nb_files_test[k] for k, v in class_nb_files.items()}

        files_test = {k: v[:class_nb_files_test[k]] for k, v in files_picked.items()}
        files_train = {k: v[class_nb_files_test[k]:] for k, v in files_picked.items()}

        move_preproc_images(files_test, os.path.join(target_dir, 'test'))
        move_preproc_images(files_train, os.path.join(target_dir, 'train'))
    else:
        move_preproc_images(files_picked, target_dir)


# from <project_root>/smearly we can run `python -m tools.image_files healthy 30 rubbish 50`

def main():
    if len(sys.argv[1:]) % 2 != 0 or len(sys.argv[1:]) == 0:
        print('I only take an even number of arguments like: dirname1 nb1 dirname2 nb2 (...)', file=sys.stderr)

    args = sys.argv[1:]
    class_nb_files = {dir: int(nb) for dir, nb in zip(args[::2], args[1::2])}

    try:
        generate_new_img_dir(class_nb_files)
        return 0
    except Exception as e:
        print(e)
        return -1

if __name__ == "__main__":
    sys.exit(main())
