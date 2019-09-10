from HOG import Hog
import misc
import os
from tqdm import tqdm

hog = Hog(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

imagePath = 'Hand_0000002.jpg'
image = misc.read_image(os.path.join(os.path.dirname(__file__), imagePath))
image_gray = misc.convert2gray(image)
misc.plot_image(image_gray)
image_input = misc.resize_image(image_gray, (120, 160))

result = hog.compute_hog(image_input)

print(result)
print(len(result))

