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

##Files:

  - `ObjectDetectionNet`:
    - CNN network to use the Regression tp detect object.

  - `Dataset`:
    - Splits the data into training, validation and test set and prepare the data.

  - `HOGClassifier`: 
    - Trains an SVM to detect cars and non-cars. It extracts HOGs features too.

  - `SearchClassify`: 
    - Implements a sliding window search for cars, including false positive filtering and applies the classifier to a video

  - `WindowsSearch`:
    - Implements the search over all zones and windows.

  - `Playground.ipynb`:
    - Where I run the previous two classes to train the data and test it out

  - `helpers.py`:
    - I has all the helpers methods we got in the class 


## Classification:

###Steps:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

###[Rubric](https://review.udacity.com/#!/rubrics/513/view):

#### Data Exploration

I'm using the data from udacitys `Vehicle Detection and Tracking` project. The images provided (car and non car) shoulb be placed in ../data/vehicles/ and ../data/non-vehicles/.

I'm using `Dataset` class to do the parsing and splitting data to validation and training sets. 

```python
from dataset import Dataset

dataset = Dataset()
dataset.call()
dataset.inspect()
```

```
X_train shape: (14208, 64, 64, 3)
14208 train samples
3552 test samples
8792 images of vehicles
8968 images of non vehicles
```

All images are 64x64 pixels. A third data set released by Udacity was not used here. 

In total there are `8792` images of vehicles and `8968` images of non vehicles. Thus the data is somehow balanced. 

The quantity and quality of these sample images is critical to the process. Bad quality images will make the classifier do wrong predictions.

These data are separated in training (80%) and validation sets (20%), and their order is randomized.

Check out below 5 random samples for cars and non-cars images: 

![sample_cars](./assets/sample_cars_non_cars.png)

#### Histogram of Oriented Gradients (HOG)

The HOG extractor is the heart of the method described here. It is a way to extract meaningful features of a image. It captures the general aspect of cars, not the specific details of it. It is the same as we, humans do, in a first glance, we locate the car, not the make, the plate, the wheel, or other small detail.

HOG stands for (Histogram of Oriented Gradients). Basically, it divides an image in several pieces. For each piece, it calculates the gradient of variation in a given number of orientations. Example of HOG detector — the idea is the image on the right to capture the essence of original image.

![Features](./assets/feature_extrac.png)

My code for extracting features in `HOGClassifier` class. 

I used the following params in my class (trial and error approach):

```
color_space = 'HLS'
spatial_size = (16, 16)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True
``` 

As you can see, I used HLS space and a low value of pixels_per_cell=(8,8). Using larger values of than orient=9 did not have a striking effect and only increased the feature vector. Similarly, using values larger than cells_per_block=(2,2) did not improve results.

The HOG algorithm is robust for small variations and different angles. But, on the other way, it can detect also some image that has the same general aspect of the car, but it not a car at all — the so called “False positives”.

The same could be made with a color detector, in addition to HOG detector. Because the HOG only classifier was good enough, I used it in the rest of project.

#### The SVC Classifier

Again, most of the code for this section can be found in `HOGClassifier` class.

The next step was to train a classifier. It receives the cars / non-cars data transformed with HOG detector, and returns if the sample is or is not a car.

In this case, I used a Support Vector Machine Classifier (SVC), with linear kernel, based on function SVM from scikit-learn.

To train my SVM classifier, I used all channels of images converted to HLS space. I included spatial features color features as well as all three HLS channels, because using less than all three channels reduced the accuracy considerably. 

The final feature vector has a length of `8460` elements, most of which are HOG features. 

For color binning patches of spatial_size=(16,16) were generated and color histograms were implemented using hist_bins=32 used.

After training on the training set this resulted in a validation and test accuracy of `0.9893`. It took about `110.73` secs to extract the data. To train the classifier it took `92.77 Seconds`. Please refer to the `Playground` notebook to check out my findings. 

The average time for a prediction (average over a hundred predictions) turned out to be about 3.3ms on my macbook pro with i7 processor, thus allowing a theoretical bandwidth of 300Hz.

A realtime application would therfore only feasible if several parts of the image are examined in parallel in a similar time.

Using just the L channel reduced the feature vector to about a third, while test and validation accuracy dropped to about 94.5% each.

Despite the high accuracy there is a systematic error as can be seen from investigating the false positive detections.

Here is a sample of false negatives for car images: 

![false_cars](./assets/misclassified_cars.png)

And here is a sample of false negatives for not cars images: 

![false_non_cars](./assets/misclassified_non_cars.png)


#### Sliding Window Search

Up to now, we can feed a classifier with an 64 x 64 pixels image and get a result from it: car or non-car.

In order to do this in an entire image (720 x 1280), we use a sliding window.

First, I cropped just the interest region. Then sliced the image in small frames, resized it to the right size (64x64), and applied the classification algorithm we created in `hog_classifier.py`.

In `WindowsSearch` class I handle all the logic for sliding window search.

- I segmented the image into 4 partially overlapping zones with different sliding window sizes to account for different distances.
- The window sizes are 240,180,120 and 70 pixels for each zone
- Within each zone adjacent windows have an ovelap of 75%


#### Video Implementation

## Regression

## Conclusion & Discussion

