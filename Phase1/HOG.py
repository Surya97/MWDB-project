from skimage import feature
import misc


class Hog:
    def __init__(self, orientations, pixels_per_cell, cells_per_block):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def compute_hog(self, image):
        (H, hogImage) = feature.hog(image, orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block, visualize=True)
        misc.plot_image(hogImage)
        return H
