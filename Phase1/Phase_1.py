import sys
from features_images import FeaturesImages

arguments = sys.argv[1:]

task = arguments[0]

model = arguments[1]

if task == '1':
    image_id = arguments[2]

elif task == '2':
    folder_path = arguments[2]
    features_folder = FeaturesImages(model, folder_path)
    features_folder.compute_features_images_folder()
if task == '3':
    image_id = arguments[2]
    k = arguments[3]

