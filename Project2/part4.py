import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import random



MLP1 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(100,100,100,100,100), max_iter=1,activation='relu', verbose='True', learning_rate='adaptive')
MLP2 = MLPRegressor(alpha=0.1, hidden_layer_sizes=(100,100,100,100,100), max_iter=10000,activation='logistic', verbose='True', learning_rate='adaptive')




x1 = np.array([172,180,190,200,218,231,240,268,290,320,341,370,410,450,500,560,600,629,635,640,671,688,691,691,705,721,750,781,790,790,791,795,826,838,839,855,882,911,
946,998,1038,1044,1074,1120,1129,1150,1196,1230,1250])
y1 = np.array([-240,-200,-160,-120,-75,-32,-10,-10,-33,-60,-79,-56,-32,-16,-15,-40,-70,-120,-169,-230,-201,-150,-85,-40,-76,-119,-174,-19,-71,-140,-180,-149,-291,-231,-187,-216,-240,-212,
-188,-117,107,-180,-225,-207,-135,-163,-107,-165,-230])
x1 = x1.reshape([len(x1),1])
y1 = y1.reshape([len(x1),1])

x2 = np.array([70,100,139,185,235,290,335,360,409,434,453,485,530,570,605,638,680,730,780,825,860,900,945,1000])
y2 = np.array([-50,-32,-21,-19,-15,-21,-30,-48,-59,-85,-105,-109,-111,-115,-121,-127,-132,-138,-146,-155,-160,-170,-175,-185])
x2 = x2.reshape([len(x2),1])
y2 = y2.reshape([len(x2),1])

MLP1.fit(x1,y1)
MLP2.fit(x2,y2)

test_points_number = 1000
x_test = np.reshape(np.linspace(0, 1300, test_points_number),[test_points_number,1])
y_test1 = MLP1.predict(x_test)
y_test2 = MLP2.predict(x_test)




plt.figure()
plt.plot(x1,y1,'red',markersize=40)
plt.plot(x_test,y_test1,'green',markersize=20)
plt.xlabel('x')
plt.legend(['learned','predicted'])

plt.figure()
plt.plot(x2,y2,'red',markersize=40)
plt.plot(x_test,y_test2,'green',markersize=20)
plt.xlabel('x')
plt.legend(['learned','predicted'])

plt.show()