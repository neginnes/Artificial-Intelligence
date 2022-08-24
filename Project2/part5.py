from pathlib import Path
from skimage.io import imread
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np

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


x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)
MLP = MLPClassifier(hidden_layer_sizes=([30, 30, 30]), activation='tanh', solver='adam', alpha=0.01, batch_size='auto',random_state=1, max_iter=200)
MLP.partial_fit(x_train, y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

scores = cross_val_score(MLP, x_test, y_test, cv=5)
print("Accuracy = ",scores.mean(),"with std = ", scores.std())