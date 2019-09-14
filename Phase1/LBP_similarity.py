import spearman_similarity
from LBP import LocalBinaryPatterns
import misc
from pathlib import Path
import os
import collections
from itertools import islice
from tqdm import tqdm

image = 'Hand_0008663.jpg'
folder = os.path.join(Path(os.path.dirname(__file__)).parent, "data/Test dataset")

image_input = misc.read_image(os.path.join(folder, image))
misc.plot_image(image_input)

image_input_gray = misc.convert2gray(image_input)

lbp = LocalBinaryPatterns(4, 1)

input_image_lbp = []

input_image_windows = misc.split_into_windows(image_input_gray, 100, 100)
for input_image_window in input_image_windows:
    lbp_pattern = lbp.computeLBP(input_image_window)
    if len(input_image_lbp) == 0:
        input_image_lbp = lbp_pattern
    else:
        input_image_lbp += lbp_pattern


dataset_lbp_features = misc.load_from_pickle(os.path.dirname(__file__), 'LBP')
print('lbp_features_dataset length', len(dataset_lbp_features.items()))
input_image_feature_rank = spearman_similarity.ranking(input_image_lbp)
print('input_feature_rank', input_image_feature_rank)
ranking = {}

for image_id, feature in dataset_lbp_features.items():
    image_feature_rank = spearman_similarity.ranking(feature)
    # print(image_feature_rank, input_image_feature_rank)
    correlation = spearman_similarity.correlation_coefficient(image_feature_rank, input_image_feature_rank)
    ranking[image_id] = correlation

sorted_results = collections.OrderedDict(sorted(ranking.items(), key=lambda val: val[1], reverse=True))

k = 5

top_k_items = {k: sorted_results[k] for k in list(sorted_results)[:k]}

for image_id in top_k_items.keys():
    misc.plot_similar_images(image_id, top_k_items[image_id])



