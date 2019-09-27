import numpy as np

import os
from Phase1.features_images import FeaturesImages
from pathlib import Path

def svd():
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, '/Phase1')
    import misc

    misc.load_from_pickle(os.path.dirname(__file__),'CM')
    '''
    ranking = {}
    for image_id, feature_vector in tqdm(dataset_images_features.items()):
        distance = model.similarity_fn(test_image_features, feature_vector)
        ranking[image_id] = distance
    '''


if __name__ == '__main__':
    svd()