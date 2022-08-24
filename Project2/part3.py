import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import random



MLP1 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(30,30,30), max_iter=2000,activation='identity', verbose='True', learning_rate='adaptive')
MLP2 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(30,30,30), max_iter=2000,activation='relu', verbose='True', learning_rate='adaptive')
MLP3 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(30,30,30), max_iter=2000,activation='relu', verbose='Trues', learning_rate='adaptive')
MLP4 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(30,30,30), max_iter=2000,activation='tanh', verbose='True', learning_rate='adaptive')

Range = 10
a = random.random() * Range
b = random.random() * Range
c = random.random() * Range

points_number = 2000
x1 = np.reshape(np.random.uniform(-10,10,points_number),[points_number,1])
x2 = np.reshape(np.random.uniform(-10,10,points_number),[points_number,1])
x3 = np.reshape(np.random.uniform(-10,10,points_number),[points_number,1])

y1 = a * x1 + b* x2 + c
y2 = x1 + x2 + x3 + abs(x1) + abs(x2) + abs(x2)
y3 = x1**4 + x2**3 + x3**2 + x2 + 5
y4 = np.sin(x1) + np.cos(-x2)

X1 = np.zeros((points_number,2))
X2 = np.zeros((points_number,3))
for i in range(points_number):
    for j in range(3):
        if j == 0 :
            X1[i][j] = x1[i]
            X2[i][j] = x1[i]
        elif j == 1:
            X1[i][j] = x2[i]
            X2[i][j] = x2[i]
        elif j == 2 :
            X2[i][j] = x3[i]


MLP1.fit(X1,y1)
MLP2.fit(X2,y2)
MLP3.fit(X2,y3)
MLP4.fit(X1,y4)

test_points_number = 2000
x_test1 = np.reshape(np.linspace(-20, 20, test_points_number),[test_points_number,1])
x_test2 = np.reshape(np.linspace(-20, 20, test_points_number),[test_points_number,1])
x_test3 = np.reshape(np.linspace(-20, 20, test_points_number),[test_points_number,1])

X1_test = np.zeros((test_points_number,2))
X2_test = np.zeros((test_points_number,3))
for i in range(test_points_number):
    for j in range(3):
        if j == 0 :
            X1_test[i][j] = x_test1[i]
            X2_test[i][j] = x_test1[i]
        elif j == 1:
            X1_test[i][j] = x_test2[i]
            X2_test[i][j] = x_test2[i]
        elif j == 2 :
            X2_test[i][j] = x_test3[i]

y1_test = MLP1.predict(X1_test)
y2_test = MLP2.predict(X2_test)
y3_test = MLP3.predict(X2_test)
y4_test = MLP4.predict(X1_test)

