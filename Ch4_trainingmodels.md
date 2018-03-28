# Training Models
## Linear Regression
Two ways to train a linear regression model:
- *closed-form* equation directly computers model params (the model params that minimize the cost function
- *Gradient Descend (GD)* is an iterative optimization approach that gradually tweaks model params to minimize cost function over training set 
and eventually converges on same params as closed form.

vector bullshit and linear regression model prediction in vectorized form on p108]

yhat = [hypothesis funcion, using model params, theta]*[x, the instance's feature vector] = [the transpose of theta]*[x, the instance's feature vector]

- use Root Mean Square Error (RMSE) as preformance measure for it. So to train a LR model, you need to find the value of theta that minimizes RMSE 
(although in practice you can use MSE)

### Normal equation
Closed form solution to find value that minimizes theta

#### lin regression
```python
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
```

## Gradient Descent
Generic optimization algorithm

GD tweaks parameters iteratively in order to minimize a cost function– think of being on a hill in dense fog and wanted to get down to the bottom, you would follow the steepest slopde down. GD measues the local gradient of the error functio with regards to theta. 

- *learning paramater* hyperparameter determines the sizes of the steps. Having the correct step size is important so you don't just skip over the the min point

- Issues with gradient descent are that it will find a local minimum if the gradient descent is not bowl shaped, BUT MSE cost functions for Linear Regression model happens to be a convex function– so no local minima. So GD is guaranteed to approach abritrarily close to the learning global minimum. 

- To use gradient descent you need to compute the gradient of the cost funtion with regards to each model parameter theta (subscript j) (like how much the cost function changes if you change that param just a bit). This is a *partial derivative* and it calculates the gradient at each step.

-   You can do these in batches

Equation on p117 of book 
[gives the gradient vector which points uphill, so just go in the opposite direction which is done by subtracting the gradient vector form theta]. Multiply the gradient vector by n to determine the size of the step. 

- Issues with gradient descent are that it will find a local minimum if the gradient descent is not bowl shaped, BUT MSE cost functions for Linear Regression model happens to be a convex function– so no local minima. So GD is guaranteed to approach abritrarily close to the learning global minimum. 

*An implementation of this algorithm*
```
eta = 0.1  # learning rate
n_iterations = 100
m = 100

theta = np.random.rand(2, 1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = th
```

To find a good learning rate you can us a grid search, but limit the number of iterations so the grid search can eliminate models that will take too long to converge.

#### How do you set the number of iterations?
If it is too low, it will not have reached the optimal solution when the algorithm stops, but too high of one will waste time. A solution to this is to set a large number of iterations but to interupt the algo when the gradient becomes tiny (when its norm becomes smaller than a tiny number e– this e is called the *tolerance*).

### Stochastic Gradient Descent
Stochastic Gradient Descent picks a random instance in the training set at every step and compute gradients based off only that instace (as oppsoed to all of the training data). This makes it fast because there is less data to manipulate at every step and makes it possible to train on huge training sets because only one instance needs to be in the memory at each iteration. 

However, it isn't as regular and will continue to bounce up and down, decreasing only on average. So when the algo stops, the final param values are good, but not optimal.

120
