from helpers import *
import numpy as np
import time
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler

class WindowsSearch():

    def __init__(self, svc, X_scaler):
        self.svc            = svc
        self.X_scaler       = X_scaler

        self.color_space    = 'HLS'
        self.spatial_size   = (16, 16)
        self.hist_bins      = 32
        self.orient         = 9
        self.pix_per_cell   = 8
        self.cell_per_block = 2
        self.hog_channel    = 'ALL'
        self.spatial_feat   = True
        self.hist_feat      = True
        self.hog_feat       = True

        self.X_start_stop   = [[None,None],[None,None],[None,None],[None,None]]
        self.XY_window      = [(240, 240), (180, 180), (120, 120), (70, 70)]
        self.XY_overlap     = [(0.75, 0.75), (0.75, 0.75), (0.75, 0.75), (0.75, 0.75)]
        self.Y_start_stop   = [[380, 500.0], [380, 470.0], [395, 455.0], [405, 440.0]]


    def search_all_frames(self, image):
        hot_windows = []
        all_windows = []

        for i in range(len(self.Y_start_stop)):
            windows = slide_window(image,
                        x_start_stop=self.X_start_stop[i],
                        y_start_stop=self.Y_start_stop[i],
                        xy_window=self.XY_window[i],
                        xy_overlap=self.XY_overlap[i])

            all_windows += [windows]

            hot_windows +=  search_windows(image, windows,
                    self.svc,
                    self.X_scaler,
                    color_space=self.color_space,
                    spatial_size=self.spatial_size,
                    hist_bins=self.hist_bins,
                    orient=self.orient,
                    pix_per_cell=self.pix_per_cell,
                    cell_per_block=self.cell_per_block,
                    hog_channel=self.hog_channel,
                    spatial_feat=self.spatial_feat,
                    hist_feat=self.hist_feat,
                    hog_feat=self.hog_feat)

        return hot_windows,all_windows

