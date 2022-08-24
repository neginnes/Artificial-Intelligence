import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

Cs = [0.01,10,10000]
i = 0
for C in Cs:
    i = i + 1
    plt.subplot(1,3,i)
    plt.ylabel("C = "+ str(C) )
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    np.random.seed(0)
    X = np.random.randn(200, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # fit the model
    clf = svm.SVC(kernel="poly",gamma=0.1 , C = C)
    clf.fit(X, Y)

    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
plt.show()
