from LBP import LocalBinaryPatterns
from pathlib import Path
import misc
import os
from tqdm import tqdm


def print_array(A):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in A]))


lbp = LocalBinaryPatterns(4, 1)
imagePath = 'Hand_0000002.jpg'
image = misc.read_image(os.path.join(os.path.dirname(__file__), imagePath))
image_gray = misc.convert2gray(image)
misc.plot_image(image_gray)
windows = misc.split_into_windows(image_gray, 100, 100)
result = []
lbp_patterns = []
count = 0
for window in windows:
    # if count >= 1:
    #     break
    # print("Window ----------------")
    # print_array(window)
    # print("--------------------------------------")
    lbp_pattern = lbp.computeLBP(window)
    # print(lbp_pattern)
    # print("pattern ------------------------------------")
    # print('****************', len(lbp_pattern))
    # print("***********************************************")
    if len(result) == 0:
        result = lbp_pattern
    else:
        result += lbp_pattern
# print('Result shape', len(result), len(result[0]))
# print(result[0])
# print(np.array(result))
print(len(result))
print(result)
