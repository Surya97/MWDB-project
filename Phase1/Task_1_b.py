from Phase1.LBP import LocalBinaryPatterns
import misc


lbp = LocalBinaryPatterns(24, 8)

image = misc.read_image("data/Hands/Hand_0000002.jpg", gray=True)
misc.plot_image(image)
image = misc.convert2gray(image)
lbp_image = lbp.computeLBP(image)
misc.plot_image(lbp_image)

