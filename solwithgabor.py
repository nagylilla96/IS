import numpy as np

from PIL import Image
from skimage import img_as_float, exposure, filters
from sklearn import svm, metrics, model_selection
from random import shuffle
from collections import namedtuple
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

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

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(np.array(kernel)), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

kernel_params = []    
results = []
for theta in range (0, 1):
	theta = (theta / 4.) * np.pi
	frequency = 0.1
	while (frequency < 0.4):
	    kernel = gabor_kernel(frequency, theta=theta)
	    params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
	    kernel_params.append(params)
	    # Save kernel and the power image for each image
	    results.append((kernel, [power(img, kernel) for img in X]))
	    frequency = frequency + 0.1

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