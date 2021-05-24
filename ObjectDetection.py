import cv2
import numpy as np
import os
import sklearn
from sklearn import svm

def createSVM():
    """ Function used to load and create and SVM model
    - Takes no parameters
    - Returns an untrained SVM machine for classification"""

    svm = sklearn.svm.SVC(kernel='rbf')
    # Above creates an SVM with a guassian kernel
    return svm

def loadData():
    """ Function to load the images and create the data set needed
        Each folder of a specific car is assigned a number which represents what label it belongs to
        -Takes no input parameters
        - Returns two matrices X and y which is the data and the correct labels. """

    dir = 'D:\\MScCourseworks\\ComputerVision\\CarDatabase\\CarDatabase'
    subdirs = [x[0] for x in os.walk(dir)]
    # Getting all the subfolders in the folder containing all images

    count = 0
    # Creating a counter variable to know the correct image class when looping over the files
    X_harris = []
    # Creating an empty array to hold the image features using harris detection
    X_HOG = []
    # Creating an empty array to hold the image features using HOG Detection
    y = []
    # Creating an empty array to hold the true labels of the image

    for subdir in subdirs:
        # Looping through all the subdirectories
        files = os.walk(subdir).__next__()[2]
        # Getting all the files from the subdirectory in question
        if (len(files)) > 0:
            for i in range(10):
                file = files[i]
                #Looping through all files in this subdirectory
                fullname = os.path.join(dir,subdir)
                fullname = os.path.join(fullname, file)
                # The above appends the filepaths so that they can be properly read.

                image = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (70, 140))
                # reading and resizing the image
                hog = cv2.HOGDescriptor()
                # Creating a hog descriptor
                hog_features = hog.compute(image)
                # Computing the HOG features

                image = np.float32(image)
                # Changing the image datatype so harris features can be extracted

                harris_features_of_image = cv2.cornerHarris(image, 2,3,0.04)
                # Computing the harris features

                # The detected features are inserted into their corresponding array
                # The correct label is also insert to the label array
                X_harris.append(harris_features_of_image)
                X_HOG.append(hog_features)
                y.append(count)


        count = count + 1
        # This increments count whenever a new subfolder is encounted as a new subfolder means a new class label

    X_HOG = np.array(X_HOG,dtype=object)
    X_harris = np.array(X_harris, dtype=object)
    y = np.array(y)
    # The above two lines ensure that the data structures are NP arrays so that specific functions can be applied.

    return X_HOG, X_harris, y

def Shuffle_and_Split(X,y):
    """ Function to randomoze the data and then split it into training and testing portions
        Inputs: X and Y which is the data
        Returns: X_Train, Y_Train, X_Test and Y_Test which are the split data after shuffling"""

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 42)
    # This shuffles the data and splits it into training and testing portions

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    model = createSVM()
    # This creates the SVM model
    X_HOG, X_harris, y = loadData()
    # This loads the data as necessary

    X_train_HOG, X_test_HOG, y_train_HOG, y_test_HOG = Shuffle_and_Split(X_HOG,y)
    # This shuffles and splits the training data used when classifying using HOG features

    nsamples_training, nx_train, ny_train = X_train_HOG.shape
    training_data_2D = X_train_HOG.reshape((nsamples_training, nx_train * ny_train))
    nsamples_testing, nx_test, ny_test = X_test_HOG.shape
    testing_data_2D = X_test_HOG.reshape((nsamples_testing, nx_test * ny_test))
    # The above four lines reshape the data so that the model can be properly fitted and trained

    model.fit(training_data_2D,y_train_HOG)
    predictions_HOG = model.predict(testing_data_2D)
    # FItting and predicting using HOG features

    accuracy_HOG = sklearn.metrics.accuracy_score(y_test_HOG, predictions_HOG)
    print(accuracy_HOG)
    scores_HOG =  sklearn.metrics.precision_recall_fscore_support(y_test_HOG, predictions_HOG)
    print(scores_HOG)
    # Computing the metrics for HOG

    # This next section does the same thing as above but uses the Harris features
    X_train_Harris, X_test_Harris, y_train_Harris, y_test_Harris = Shuffle_and_Split(X_harris, y)
    # This shuffles and splits the training data used when classifying using HOG features

    nsamples_training, nx_train, ny_train = X_train_Harris.shape
    training_data_2D = X_train_Harris.reshape((nsamples_training, nx_train * ny_train))
    nsamples_testing, nx_test, ny_test = X_test_Harris.shape
    testing_data_2D = X_test_Harris.reshape((nsamples_testing, nx_test * ny_test))
    # The above four lines reshape the data so that the model can be properly fitted and trained

    model.fit(training_data_2D, y_train_Harris)
    predictions_harris = model.predict(testing_data_2D)
    # Fitting and predicting

    accuracy_Harris = sklearn.metrics.accuracy_score(y_test_Harris, predictions_harris)
    print(accuracy_Harris)
    scores_Harris = sklearn.metrics.precision_recall_fscore_support(y_test_HOG, predictions_harris)
    print(scores_Harris)
    # Computing The evaluation metrics




