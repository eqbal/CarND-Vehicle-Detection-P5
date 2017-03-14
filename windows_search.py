from helpers import *
import numpy as np
import time
from scipy.ndimage.measurements import label

class WindowsSearch():

    def __init__(self, svc, X_scaler):
        self.svc            = svc
        self.X_scaler       = X_scaler

        self.color_space    = 'HLS'
        self.spatial_size   = (16, 16)
        self.hist_bins      = 16
        self.orient         = 9
        self.pix_per_cell   = 8
        self.cell_per_block = 2
        self.hog_channel    = 'ALL'
        self.spatial_feat   = True
        self.hist_feat      = True
        self.hog_feat       = True
        self.cells_per_step = 1

        self.all_heats      = []

        self.scales         = [1, 1.5, 2, 2.5, 4]
        self.window         = 64
        self.y_start_stops  = [[380, 460], [380, 560], [380, 620], [380, 680], [350, 700]]

    def convert_color(self, img, color_space='RGB'):
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)

        return feature_image


    def find_cars(self, img):

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        hot_windows = []

        for y_start_stop, scale in zip(self.y_start_stops, self.scales):

            img_tosearch = img[y_start_stop[0]:y_start_stop[1],:,:]

            ctrans_tosearch = self.convert_color(img_tosearch, color_space=self.color_space)

            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                             (np.int(imshape[1]/self.scale), np.int(imshape[0]/self.scale)))

            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
            nyblocks = (ch1.shape[0] // self.pix_per_cell)-1

            nfeat_per_block = self.orient*self.cell_per_block**2

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            nblocks_per_window = (self.window // self.pix_per_cell)-1

            nxsteps = (nxblocks - nblocks_per_window) // self.cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // self.cells_per_step

            i = 0

            for xb in range(nxsteps+1):
                for yb in range(nysteps+1):
                    i += 1

                    if xb == (nxsteps + 1):
                        xpos = ch1.shape[1] - nblocks_per_window
                    else:
                        xpos = xb*self.cells_per_step

                    if yb == (nysteps + 1):
                        ypos = ch1.shape[0] - nblocks_per_window
                    else:
                        ypos = yb*self.cells_per_step


                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                    if hog_channel == 'ALL':
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    elif hog_channel == '0':
                        hog_features = hog_feat1
                    elif hog_channel == '1':
                        hog_features = hog_feat2
                    elif hog_channel == '2':
                        hog_features = hog_feat3

                    xleft = xpos*self.pix_per_cell
                    ytop  = ypos*self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(
                            ctrans_tosearch[ytop:ytop+self.window, xleft:xleft+self.window], (64,64))

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    hist_features    = color_hist(subimg, nbins=self.hist_bins)

                    img_features = []

                    if self.spatial_feat:
                        img_features.append(spatial_features)
                    if self.hist_feat:
                        img_features.append(hist_features)
                    if self.hog_feat:
                        img_features.append(hog_features)

                    img_features = np.concatenate(img_features).reshape(1, -1)

                    # Scale features and make a prediction
                    test_features   = self.X_scaler.transform(img_features)
                    test_prediction = self.svc.predict(test_features)

                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)

                    if test_prediction == 1:
                        hot_windows.append((
                            (xbox_left, ytop_draw+y_start_stop[0]),
                            (xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0])))

        return hot_windows
