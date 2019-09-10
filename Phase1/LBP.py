from skimage import feature
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def computeLBP(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints, R=self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=100,
                                 range=(0.0, 255.0))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # print('Len hist to list', len(hist.tolist()))
        # return the histogram of Local Binary Patterns
        return hist.tolist()

