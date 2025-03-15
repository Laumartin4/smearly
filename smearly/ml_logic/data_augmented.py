import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

print('start')
datagen = ImageDataGenerator(
    horizontal_flip=True, # Retournement horizontal
    vertical_flip=True, #Retournement vertical
    fill_mode='constant',    # Remplissage des pixels vides
    cval=0
)


input_folder = "raw_data/Train Data Dir/unhealthy"
output_folder = "raw_data/Train Data Dir/unhealthy augmented"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Liste des images de la classe
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

num_generated = 0  # Compteur d'images générées
num_target = 5814  # Objectif : générer num_target nouvelles images

for img_name in image_files:
    if num_generated >= num_target:
        break

    img_path = os.path.join(input_folder, img_name)
    image = load_img(img_path)  # Charger l'image
    img_array = img_to_array(image)  # Convertir en tableau numpy
    img_array = img_array.reshape((1,) + img_array.shape)  # Redimensionner pour ImageDataGenerator

    # Générer et sauvegarder de nouvelles images
    i = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_folder, save_prefix=img_name[-15:-3]+'_aug', save_format='jpg'):
        i += 1
        num_generated += 1
        if num_generated >= num_target or i >= 1:
            break

print(f"✅ {num_generated} nouvelles images enregistrées dans {output_folder}")
