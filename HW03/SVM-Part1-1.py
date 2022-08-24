import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles

cluster_stds = [0.1,0.3,0.7]
Cs = [0.01,10,10000]
i = 0
for cluster_std in cluster_stds:
    for C in Cs:
        i = i + 1
        plt.subplot(3,3,i)
        plt.ylabel("std = "+ str(cluster_std) + ", C = "+ str(C))
        # we create 40 separable points
        #X, y = make_blobs(n_samples= 200, centers=2, random_state=5, cluster_std = cluster_std )
        X, y = make_circles(n_samples=200, random_state=5, noise=0.3, factor = cluster_std)

        # fit the model, don't regularize for illustration purposes
        clf = svm.SVC(kernel="poly", gamma = 1, C = C)
        clf.fit(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model


        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(
            XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
        )
        # plot support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )
plt.show()