from LBP import LocalBinaryPatterns
from pathlib import Path
import misc
import os
from tqdm import tqdm


lbp = LocalBinaryPatterns(24, 3)
folder = os.path.join(Path(os.path.dirname(__file__)).parent, "data/Test dataset")
files_in_directory = misc.get_images_in_directory(folder)
dict_file_pattern = {}
count = 0
print('Files in the directory', len(files_in_directory.items()))
lbp_patterns = []
for file, path in tqdm(files_in_directory.items()):
    result = []
    try:
        image = misc.read_image(path)
        image_gray = misc.convert2gray(image)
        windows = misc.split_into_windows(image_gray, 100, 100)
        for window in windows:
            lbp_pattern = lbp.computeLBP(window)
            if len(result) == 0:
                result = lbp_pattern
            else:
                result += lbp_pattern
        dict_file_pattern[file] = result
        lbp_patterns.append(result)
    except:
        lbp_patterns.append(result)

# bar.finish()
print(len(list(files_in_directory.keys())), len(lbp_patterns))
images = list(files_in_directory.keys())
folder_features_dict = {}
for i in range(len(images)):
    folder_features_dict[images[i]] = lbp_patterns[i]

# misc.save2csv(zip(list(files_in_directory.keys()), lbp_patterns), os.path.dirname(__file__), feature='LBP')
misc.save2pickle(folder_features_dict, os.path.dirname(__file__), feature='LBP')

