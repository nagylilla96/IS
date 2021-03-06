import numpy as np
import time
from PIL import Image
from skimage import img_as_float, exposure
from sklearn import svm, metrics, model_selection
from random import shuffle
from collections import namedtuple
from sklearn.decomposition import PCA

start_time = time.time()

size = 920, 640

healthy = []
glaucoma = []
diabet = []

# read dataset into an array of images and return it
# process the images by:
# 	- converting them to black & white
# 	- resizing them to the same size (240x160)
# 	- normalize the contrast (with percentile and rescale_intensity) 
def readdataset(dataset, folder, tag, nr):
	for x in range(1,nr):
		img = Image.open('./everything/' + folder + '/' + '{0:02}'.format(x) + tag + '.jpg')
		img = img.resize(size, 0)
		image = np.array(img)
		p2, p98 = np.percentile(image, (2, 98))
		img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
		dataset.append(img_rescale)
	return dataset

# read the two datasets from file
healthy = readdataset(healthy, 'healthy', '_h', 16)
glaucoma = readdataset(glaucoma, 'glaucoma', '_g', 16)
diabet = readdataset(diabet, 'diabetic_retinopathy', '_dr', 16)
print("Data set read")

# read the labels
hlabel = ["healthy"]*15
glabel = ["glaucoma"]*15
drlabel = ["diabetic_retinopathy"]*15

X = np.concatenate((np.array(healthy).reshape((15, -1)), np.array(glaucoma).reshape((15, -1)),
	np.array(diabet).reshape((15, -1))))
y = hlabel + glabel + drlabel

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=42)

print("Training sets split")

# Principal Component Analysis (PCA) - decomposes data to a lower dimension space
# Improved running time, takes out only 150 components out of the full set
n_components = 15

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Done")

# initialize the SVC(Support Vector Machine) classifier with GridSearchCV
# which will probe the values against the param-grid dictionary, as also given by
# the 'rbf' kernel (Radial Basis Function), which aids classification
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
classifier = model_selection.GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)

print("Classifier initialized")

# fit the data
classifier.fit(X_train_pca, y_train)

print("Fitting done")

# get the predicted values
y_pred = classifier.predict(X_test_pca)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))

print(metrics.confusion_matrix(y_test, y_pred))

print("Running time: %s s" % (time.time() - start_time))