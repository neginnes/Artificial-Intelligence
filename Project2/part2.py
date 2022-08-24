import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import random



MLP1 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='identity', verbose='True', learning_rate='adaptive')
MLP2 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='relu', verbose='True', learning_rate='adaptive')
MLP3 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='relu', verbose='Trues', learning_rate='adaptive')
MLP4 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='relu', verbose='True', learning_rate='adaptive')
MLP5 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='tanh', verbose='True', learning_rate='adaptive')
MLP6 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(15,15,15), max_iter=1000,activation='tanh', verbose='True', learning_rate='adaptive')

Range = 10
a = random.random() * Range
b = random.random() * Range
points_number = 1000
x = np.reshape(np.linspace(-10,10, points_number),[points_number,1])
noise = np.random.rand(points_number) - 0.5*np.ones(points_number)
y1 = a * x + b
y2 = x + abs(x)
y3 = 0.5*x**3
y4 = x**4 + x**3 + x**2 + x + 5
y5 = np.sin(x)
y6 = np.sin(x) + np.cos(-x)

noise_amount = 0.01

y1 = y1 + noise_amount*max(y1)*noise
y2 = y2 + noise_amount*max(y2)*noise
y3 = y3 + noise_amount*max(y3)*noise
y4 = y4 + noise_amount*max(y4)*noise
y5 = y5 + noise_amount*max(y5)*noise
y6 = y6 + noise_amount*max(y6)*noise

MLP1.fit(x,y1)
MLP2.fit(x,y2)
MLP3.fit(x,y3)
MLP4.fit(x,y4)
MLP5.fit(x,y5)
MLP6.fit(x,y6)

test_points_number = 2000
x_test = np.reshape(np.linspace(-20, 20, test_points_number),[test_points_number,1])
y1_test = MLP1.predict(x_test)
y2_test = MLP2.predict(x_test)
y3_test = MLP3.predict(x_test)
y4_test = MLP4.predict(x_test)
y5_test = MLP5.predict(x_test)
y6_test = MLP5.predict(x_test)

plt.figure()
plt.subplot(3, 2, 1)
plt.plot(x,y1,'green',markersize=40)
plt.plot(x_test,y1_test,'red',markersize=20)
plt.ylabel('ax+b (a,b are random numbers)')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')

plt.subplot(3, 2, 2)
plt.plot(x,y2,'green',markersize=40)
plt.plot(x_test,y2_test,'red',markersize=20)
plt.ylabel('x+|x|')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')

plt.subplot(3, 2, 3)
plt.plot(x,y3,'green',markersize=40)
plt.plot(x_test,y3_test,'red',markersize=20)
plt.ylabel('0.5 x^3')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')

plt.subplot(3, 2, 4)
plt.plot(x,y4,'green',markersize=40)
plt.plot(x_test,y4_test,'red',markersize=20)
plt.ylabel('x^4 + x^3 + x^2 + x + 5')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')

plt.subplot(3, 2, 5)
plt.plot(x,y5,'green',markersize=40)
plt.plot(x_test,y5_test,'red',markersize=20)
plt.ylabel('sin(x)')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')

plt.subplot(3, 2, 6)
plt.plot(x,y6,'green',markersize=40)
plt.plot(x_test,y6_test,'red',markersize=20)
plt.ylabel('sin(x)+cos(-x)')
plt.xlabel('x')
plt.legend(['learned','predicted'], loc = 'upper left')
plt.show()