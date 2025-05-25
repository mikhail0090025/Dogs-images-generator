import os

images_shape = (100,100, 3)
images_resolution = (images_shape[0], images_shape[1])
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'dog-and-cat-classification-dataset.zip'
dataset_directory = os.path.join(current_dir, 'PetImages')
