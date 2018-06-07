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

# read the labels
blabel = ["background"]*449
rlabel = ["raccoon"]*140
learnlabel = []

# prepare the learnset and the testset(flatten the images to 2 dimensions and concatenate them)
# data1 = np.array(backgrounds[:229]).reshape((229, -1))
# data2 = np.array(raccoons[:70]).reshape((70, -1))
# data3 = np.array(backgrounds[229:]).reshape((220, -1))
# data4 = np.array(raccoons[70:]).reshape((70, -1))

X = np.concatenate((np.array(backgrounds).reshape((449, -1)), np.array(raccoons).reshape((140, -1))))
y = blabel + rlabel

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=42)

# learnset = np.concatenate((data1,data2))
# random_learn =  zip(learnset, blabel[:229] + rlabel[:70])
# shuffle(random_learn)
# learnset, learnlabel = zip(*random_learn)
# testset = np.concatenate((data3,data4))

# initialize the svc classifier
classifier = svm.SVC(C=1.0, kernel='linear', class_weight='balanced')

# fit the data
classifier.fit(X_train, y_train)

# prepare the expected label list
expected = blabel[229:] + rlabel[70:]

# get the predicted values
predicted = classifier.predict(testset)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

print(metrics.confusion_matrix(expected, predicted))