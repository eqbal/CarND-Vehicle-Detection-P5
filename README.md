# Vehicle Detection Project

#### By: [Eqbal Eki](http://www.eqbalq.com/)

****
This project goal is to write a software pipeline to detect vehicles in a video.  This is a project for 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive).

We can use one of the two approaches to solve this problem:

- **Classification** problem: 

  The image are divided into small patches, each of which will be run through a classifier to determine whether there are objects in the patch. Then the bounding boxes will be assigned to locate around patches that are classified with high probability of present of an object. The steps for this approaches are:

  - A Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
  - A color transform and binned color features, as well as histograms of color, to combine the HOG feature vector with other classical computer vision approaches
  - A sliding-window technique to search for cars with the trained SVM
  - Creating a heatmap of recurring detections in subsequent framens of a video stream to reject outliers and follow detected vehicles.

- **Regression** problem using [YOLO](https://pjreddie.com/darknet/yolo/).

  Here, the whole image will be run through a convolution neural network (CNN) to directly generate one or more bounding boxes for objects in the images. The steps can be briefed below:

  - Regression on the whole image to generate bounding boxes.
  - Generate bounding box coordinates directly from CNN

In this project, I'll be using both techniques, in Classification one, we will be able to lower the false positives further. Once I'm done, I'll be using Regression with YOLO and compare the results.
****

## Classification:

###Files:

  - `HOGClassifier`: 
    - Splits the data into training, validation and test set and saves them in a pickle file.
    - Trains an SVM to detect cars and non-cars. All classifier data is saved in a pickle file.

  - `SearchClassify`: 
    - Implements a sliding window search for cars, including false positive filtering and applies the classifier to a video

  - `Playground.ipynb`:
    - Where I run the previous two classes to train the data and test it out

  - `helpers.py`:
    - I has all the helpers methods we got in the class 

###Steps:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

###[Rubric](https://review.udacity.com/#!/rubrics/513/view):

#### Data Exploration

#### Histogram of Oriented Gradients (HOG)

#### Sliding Window Search

#### Video Implementation

## Regression

## Conclusion & Discussion

