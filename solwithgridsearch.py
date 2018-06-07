import numpy as np

from PIL import Image
from skimage import img_as_float, exposure
from sklearn import svm, metrics, model_selection
from random import shuffle
from collections import namedtuple

size = 240, 160

backgrounds = []
raccoons = []

# read dataset into an array of images and return it
# process the images by:
# 	- converting them to black & white
# 	- resizing them to the same size (240x160)
# 	- normalize the contrast (with percentile and rescale_intensity) 
def readdataset(dataset, key, nr):
	for x in range(1,nr):
		img = Image.open('./everything/' + key + '_0' + '{0:03}'.format(x) + '.jpg').convert('LA')
		img = img.resize(size, 0)
		image = np.array(img)
		p2, p98 = np.percentile(image, (2, 98))
		img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
		dataset.append(img_rescale)
	return dataset

# read the two datasets from file
backgrounds = readdataset(backgrounds, 'background/257', 450)
raccoons = readdataset(raccoons, 'raccoons/168', 141)
print("Data set read")

# read the labels
blabel = ["background"]*449
rlabel = ["raccoon"]*140

X = np.concatenate((np.array(backgrounds).reshape((449, -1)), np.array(raccoons).reshape((140, -1))))
y = blabel + rlabel

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=42)

print("Training sets split")

# initialize the svc classifier
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
classifier = model_selection.GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)

print("Classifier initialized")

# fit the data
classifier.fit(X_train, y_train)

print("Fitting done")

# prepare the expected label list
expected = blabel[229:] + rlabel[70:]

# get the predicted values
y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))

print(metrics.confusion_matrix(y_test, y_pred))