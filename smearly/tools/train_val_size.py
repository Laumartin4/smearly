# Description: This file contains the function to split the dataset into training and validation sets
import os

def train_val_size(
    val_distribution: list[float, int], # [rubbish, healthy, unhealthy]
    train_distribution: list[float, int], # [rubbish, healthy, unhealthy]
    path: str = '../raw_data/all/unhealthy_bothcells_augmented',
    test_val_split: float = 0.3) -> dict:
    """
    Function to define the train and validation repartition with the desired number of images per class
    """

    # Convertir les pourcentages en float (si ce sont des entiers)
    train_distribution = [float(x) for x in train_distribution]
    val_distribution = [float(x) for x in val_distribution]

    # Vérifier que les distributions totalisent 1.0 (ou 100%)
    if sum(train_distribution) != 1.0 and sum(train_distribution) != 100.0:
        raise ValueError("Train distribution must sum to 1.0 or 100.")
    if sum(val_distribution) != 1.0 and sum(val_distribution) != 100.0:
        raise ValueError("Validation distribution must sum to 1.0 or 100.")

    # Normaliser les pourcentages si nécessaire (si total = 100)
    if sum(train_distribution) > 1.0:
        train_distribution = [x / 100.0 for x in train_distribution]
    if sum(val_distribution) > 1.0:
        val_distribution = [x / 100.0 for x in val_distribution]

    unhealthy_bothcells_augmented_and_unhealthy_bothcells_size = 2*len(os.listdir(path))

    total_size = unhealthy_bothcells_augmented_and_unhealthy_bothcells_size/train_distribution[2]
    unhealthy_bothcells_augmented_and_unhealthy_bothcells_size , total_size
    
    test_val_split = test_val_split

    val_size = total_size*test_val_split
    train_size = total_size - val_size

    train_unhealthy_size = train_distribution[2]*train_size
    train_rubbish_size = train_distribution[0]*train_size
    train_healthy_size = train_distribution[1]*train_size
    val_unhealthy_size = val_size*val_distribution[2]
    val_rubbish_size = val_size*val_distribution[0]
    val_healthy_size = val_size*val_distribution[1]
    
    set_class_nb_files = {
    'train': {
        'bothcells': 0,
        'healthy': round(train_healthy_size),
        'rubbish': round(train_rubbish_size),
        'unhealthy': 0,
        'unhealthy_bothcells': round(train_unhealthy_size/2),
        'unhealthy_bothcells_augmented': round(train_unhealthy_size/2)
    },
   
   #must be different than train set
   
    'val': {
        'bothcells': 0,
        'healthy': round(val_healthy_size),
        'rubbish': round(val_rubbish_size),
        'unhealthy': 0,
        'unhealthy_bothcells': round(val_unhealthy_size), 
        'unhealthy_bothcells_augmented': 0
    }
    }
    
    return set_class_nb_files
