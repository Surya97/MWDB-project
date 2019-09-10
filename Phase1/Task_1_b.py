from Phase1.LBP import LocalBinaryPatterns
from pathlib import Path
import misc
import os
from tqdm import tqdm
import progressbar
from time import sleep
import numpy

lbp = LocalBinaryPatterns(8, 2)
folder = os.path.join(Path(os.path.dirname(__file__)).parent, "data/Hands")
files_in_directory = misc.get_images_in_directory(folder)

# bar = progressbar.ProgressBar(maxval=len(files_in_directory.items()),
#                               widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

dict_file_pattern = {}
count = 0
print('Files in the directory', len(files_in_directory.items()))
lbp_patterns = []
# bar.start()
for file, path in tqdm(files_in_directory.items()):
    result = []
    try:
        image = misc.read_image(path)
        image_gray = misc.convert2gray(image)
        windows = misc.split_into_windows(image_gray, 100, 100)
        # print(windows)
        for window in windows:
            lbp_pattern = lbp.computeLBP(window)
            # print(lbp_pattern)
            if len(result) == 0:
                result = lbp_pattern
            else:
                result += lbp_pattern

        # print('Result shape', len(result), len(result[0]))
        dict_file_pattern[file] = result
        lbp_patterns.append(result)
    except:
        lbp_patterns.append(result)

# bar.finish()
print(len(list(files_in_directory.keys())), len(lbp_patterns))
misc.save2csv(list(files_in_directory.keys()), lbp_patterns, os.path.dirname(__file__), feature='LBP')


