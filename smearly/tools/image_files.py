import os
import sys
import random
import shutil
from tensorflow.image import encode_png as tf_encode_png
from tensorflow.io import write_file as tf_write_file

from smearly.ml_logic.preprocessing import image_file_to_tf, resize_pad_image_tf

def generate_new_img_dir(
    set_class_nb_files: dict[str, dict[str,int]],
    resize_pad_size: tuple[int,int] | None = (224, 224),
    all_img_basedir: str = '../raw_data/all',
    target_dir: str = '../raw_data/reduced',
    random_seed: int | None = 42
    ) -> None:
    """
    Take some images from subdirs of `all_img_basedir` and put them in `target_dir`.
    Optionally create `target_dir`/(test & train) if test_size is not None.

    Args:
    - class_nb_files: a dictionary with keys being subdir names, and values being
      the number of image files to take, like:
    ```python
    set_class_nb_files = {
        'train': {
            'bothcells': 30,
            'healthy': 10,
            'unhealthy': 10,
            'rubbish': 10
        },
        'test': {
            'bothcells': 6,
            'healthy': 2,
            'unhealthy': 2,
            'rubbish': 2
        },
        'val': {
            'bothcells': 0.01,
            'healthy': 0.01,
            'unhealthy': 0.01,
            'rubbish': 0.01
        }
    }
    ```
    - resize_pad_size (tuple or None): Target size (height, width) of the output
      image (default is 224x224) or None to keep the original image (simple copy).
    - all_img_basedir (str): path to the dir containing all images
    - target_dir (str): path to the dir where generated/moved images will be put
    - random_seed (int or None): set to constant to always get the same images
      list with the same class_nb_files

    Returns:
    - nothing (None)
    """

    subdirs = [f for f in os.listdir(all_img_basedir) if os.path.isdir(os.path.join(all_img_basedir, f))]

    if random_seed is not None:
        random.seed(random_seed)

    # make a dict of the *total* number of source files to pick per class, all sets included (train, val, test)
    # {'rubbish': 12, 'healthy': 47, ...}
    class_total_nb_files = {}
    for set_name, class_nb_files in set_class_nb_files.items():
        for class_name, nb_files_requested in class_nb_files.items():
            if not isinstance(nb_files_requested, (int, float)):
                print(f'[Error!] got {nb_files_requested} for {class_nb_files}, {class_name} but expecting int or float.', file=sys.stderr)
                return None
            if isinstance(nb_files_requested, float):
                if not (0. <= nb_files_requested <= 1.):
                    print(f'[Error!] when using a float, it must be between 0 and 1, got {nb_files_requested} for {class_nb_files}, {class_name}', file=sys.stderr)
                    return None
                nb_files_requested = int(nb_files_requested * sum(1 for entry in os.scandir(os.path.join(all_img_basedir, class_name)) if entry.is_file()))
                set_class_nb_files[set_name][class_name] = nb_files_requested


            class_total_nb_files[class_name] = class_total_nb_files.get(class_name, 0) + nb_files_requested

    # make a dict of *all* the *source* files_names to pick, per class, all sets combined (train, test, val...)
    src_files_picked = {}
    for subdir in subdirs:
        if subdir in class_total_nb_files:
            file_names = [f for f in os.listdir(os.path.join(all_img_basedir, subdir)) if os.path.isfile(os.path.join(all_img_basedir, subdir, f))]
            nb_files_to_pick = class_total_nb_files[subdir]
            nb_file_names = len(file_names)
            if nb_files_to_pick > nb_file_names:
                print(f'[Warning!] You asked for {nb_files_to_pick} files in {subdir} which contains only {nb_file_names} file(s)', file=sys.stderr)
                print(f'Reducing number of picked files to the maximum: {nb_file_names} file(s)\n', file=sys.stderr)
                nb_files_to_pick = nb_file_names

            src_files_picked[subdir] = random.sample(sorted(file_names), nb_files_to_pick)

    def move_preproc_images(src_files_picked: dict[str:list[str]], target_dir: str):
        """
        Move files from `all_img_basedir` to `target_dir` according to the `src_files_picked` dictionary

        Args:
         - src_files_picked: source files in dict like {'class_name (unhealthy, rubbish...)': ['file1.png', 'file2.png']}
         - target_dir: where to copy them (this function will auto-create one subdir for each class_name)
        """
        for dir, file_names in src_files_picked.items():
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

    # for each set, move files within their class
    for set_name, class_nb_files in set_class_nb_files.items():

        # build class_files dict (like {'class_name': ['file1.png', 'file2.png']}) for this set
        class_files = {}
        for src_class, src_files in src_files_picked.items():
            if src_class in class_nb_files:
                # get the first N files from src_files. N as requested by user.
                class_files[src_class] = src_files[:class_nb_files[src_class]]

                # remove the N first files from files_picked file list, to avoid re-picking them later
                src_files_picked[src_class] = src_files_picked[src_class][class_nb_files[src_class]:]

        # move files for the current set
        move_preproc_images(class_files, os.path.join(target_dir, set_name))


# NOTE function above changed, command line interface is not working any more
# # from <project_root>/smearly we can run `python -m tools.image_files healthy 30 rubbish 50`

# def main():
#     if len(sys.argv[1:]) % 2 != 0 or len(sys.argv[1:]) == 0:
#         print('I only take an even number of arguments like: dirname1 nb1 dirname2 nb2 (...)', file=sys.stderr)

#     args = sys.argv[1:]
#     class_nb_files = {dir: int(nb) for dir, nb in zip(args[::2], args[1::2])}

#     try:
#         generate_new_img_dir(class_nb_files)
#         return 0
#     except Exception as e:
#         print(e)
#         return -1

# if __name__ == "__main__":
#     sys.exit(main())
