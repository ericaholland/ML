import numpy as np
import matplotlib.pyplot as plt

"""generate some random linear looking data"""
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)

""" Computing  theta hat (minimizes cost function) using normal equation"""
X_b = np.c_[np.ones((100,1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # .dot is matrix multipl. linalg.inv gives inverse of matr
print(theta_best)  # what it's guessing the 4 and 3 are, so it's not that close

# using theta hat/ theta_best to make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")  # original random points

"""labeling"""
# plt.plot(X, y, "b.")  # plotted original, random points
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
# save_fig("generated_data_plot")
plt.show()
