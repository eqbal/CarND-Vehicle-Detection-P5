from helpers import *
from windows_search import *
from scipy.ndimage.measurements import label


class ImageProcessor():
    def __init__(self, win_search):
        self.win_search = win_search

    def call(self, image):
        draw_image = np.copy(image)

        heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        hot_windows = self.win_search.find_cars(image)

        heatmap = add_heat(heatmap, hot_windows)


        heatmap = apply_threshold(heatmap,2)
        heat = apply_threshold(heat,3)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # draw the bounding box on the image 
        draw_image = np.copy(image)
        draw_image = draw_labeled_bboxes(draw_image, labels)

        return draw_image

