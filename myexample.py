import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from PIL import Image

n_colors = 32

# Load the raccoon image
waschbar = Image.open("waschbar.jpg")

# Convert to floats and then divide by 255 so that imshow will work on it
# (The values have to be in range [0,1])
waschbar = np.array(waschbar, dtype=np.float64) / 255

# Transform image to a 2D array
w, h, d = original_shape = tuple(waschbar.shape)
assert d == 3
image_array = np.reshape(waschbar, (w * h, d))

# Fitting the model on a small subsample of the data
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

# Predicting color indices on the full image using k-means
labels = kmeans.predict(image_array)

# Predicting color indices on the full image randomly 
random = shuffle(image_array, random_state=0)[:n_colors + 1]
labels_random = pairwise_distances_argmin(random, image_array, axis=0)


# Recreate image of a certain size using the codebood and the labels
def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(waschbar)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (32 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (32 colors, Random)')
plt.imshow(recreate_image(random, labels_random, w, h))
plt.show()
