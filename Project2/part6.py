
from pathlib import Path
from skimage.io import imread
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from random import randint,random,sample
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def noise_adder(images,noise_ratio):
    for image in images:
        pixel_set = sample(range(0, len(image)), int(noise_ratio*len(image)))
        for pixel in pixel_set:
            if (randint(1,100) > 30):
                if (image[pixel]>128):
                    image[pixel] = randint(0,128)
                else :
                    image[pixel] = randint(128,255)
    return images

image_dir = Path("images")
folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
flat_images = []
target = []
for n, directory in enumerate(folders):
    for file in directory.iterdir():
        img = imread(file)
        name = (str(file.mkdir).split('\'')[1]).split('/')[2][0]
        flat_images.append(img.flatten())
        target.append(int(name))
flat_images = np.array(flat_images)
target = np.array(target)
dataset = Bunch(data=flat_images,target=target)



main_images = dataset.data
noise_ratio = 0.1
noisy_images = noise_adder(deepcopy(main_images), noise_ratio)
x_train, x_test, y_train, y_test = train_test_split(noisy_images, main_images, test_size=0.3)
MLP = MLPRegressor(alpha=0.1, hidden_layer_sizes=(500,500,500), max_iter=2000,activation='relu', verbose='True', learning_rate='adaptive')
MLP.fit(x_train, y_train)
print("Accuracy = ",MLP.score(x_test, y_test))          # test score
pred_y_test = MLP.predict(x_test)
pred_y_train = MLP.predict(x_test)

chosen_images = np.random.choice(np.arange(len(y_test)), size=10)
images = np.zeros((10, 3, x_test[0].shape[0]))
images[:, 0] = y_test[chosen_images]
images[:, 1] = x_test[chosen_images]
images[:, 2] = pred_y_test[chosen_images]
images = images.reshape(10 * 3, x_test[0].shape[0] )

fig1 = plt.figure(figsize=(30,30))                                          # test results figure
for i in range(30):
    ax = fig1.add_subplot(10, 3, i + 1)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(images[i].reshape(16,16))
plt.show()

chosen_images = np.random.choice(np.arange(len(y_test)), size=10)
images = np.zeros((10, 3, x_test[0].shape[0]))
images[:, 0] = y_test[chosen_images]
images[:, 1] = x_test[chosen_images]
images[:, 2] = pred_y_train[chosen_images]
images = images.reshape(10 * 3, x_test[0].shape[0] )

fig2 = plt.figure(figsize=(20,20))                                           # train results figure
for i in range(30):
    ax = fig2.add_subplot(10, 3, i + 1)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(images[i].reshape(16,16))
plt.show()
print("Accuracy = ",MLP.score(x_train, y_train))          # train score