from skimage import feature
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def computeLBP(self, image):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints, R=self.radius, method='uniform')
        return lbp
