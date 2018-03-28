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

*Comparison of algorithms for Linear Regression*

Algo| Large *m* | Out-of-core support | Large *n* | Hyperparams | Scaling required | Scikit-Learn 
|--|--|--|--|--|--|--|
|**Normal equation** | Fast | No | Slow | 0 | No | LinearRegresion|
|**Batch GD**| Slow | No | Fast| 2 | Yes | n/a |
|**Stochastics GD** | Fast | Yes | Fast | ≥2 | Yes | SGDRegressor |
|**Mini-batch GD**| Fast | Yes | Fast | ≥2 | Yes | SGDRegressor |

*m* is number of training instances and *n* is number of features. 
There is almost no difference after training; all of them end up with similar models and make predictions the same way.

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

|Batch GD|Stochastic GD|Mini-Batch GD|
|--|--|--|
|Computes gradients based on full training set|Computes based off just one random instance |Computes gradients on small random sets of instances |
||better for large sets, but slightly less accurate|better than stochastic bc you can get a preformance boost from hardware optimization|

GD tweaks parameters iteratively in order to minimize a cost function– think of being on a hill in dense fog and wanted to get down to the bottom, you would follow the steepest slopde down. GD measues the local gradient of the error functio with regards to theta. 

- *learning paramater* hyperparameter determines the sizes of the steps. Having the correct step size is important so you don't just skip over the the min point

- Issues with gradient descent are that it will find a local minimum if the gradient descent is not bowl shaped, BUT MSE cost functions for Linear Regression model happens to be a convex function– so no local minima. So GD is guaranteed to approach abritrarily close to the learning global minimum. 

- To use gradient descent you need to compute the gradient of the cost funtion with regards to each model parameter theta (subscript j) (like how much the cost function changes if you change that param just a bit). This is a *partial derivative* and it calculates the gradient at each step.

-   You can do these in batches

Equation on p117 of book 
[gives the gradient vector which points uphill, so just go in the opposite direction which is done by subtracting the gradient vector form theta]. Multiply the gradient vector by n to determine the size of the step. 

- Issues with gradient descent are that it will find a local minimum if the gradient descent is not bowl shaped, BUT MSE cost functions for Linear Regression model happens to be a convex function– so no local minima. So GD is guaranteed to approach abritrarily close to the learning global minimum. 

*An implementation of this algorithm*
```python
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

**Cons:**
However, it isn't as regular and will continue to bounce up and down, decreasing only on average. So when the algo stops, the final param values are good, but not optimal. It never settles at the true minimum

**Pros:**
If the cost fuction is irregular, it this up and down can help the algorithm jump out of a local minimum. 

To deal with this, gradually reduce the learning rate. By starting out with large steps, it won't take as long and you escape local minima, as they get smaller is settles on the global minimum. This is *simulated annealing*. The function that determines the learning rate is the *learning schedule*.

*Implementing Stochastic Gradient Descent using a simple learning schedule:*
```python
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # not shown in the book
            y_predict = X_new_b.dot(theta)           # not shown
            style = "b-" if i > 0 else "r--"         # not shown
            plt.plot(X_new, y_predict, style)        # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 # not shown

plt.plot(X, y, "b.")                                 # not shown
plt.xlabel("$x_1$", fontsize=18)                     # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
plt.axis([0, 2, 0, 15])                              # not shown
save_fig("sgd_plot")                                 # not shown
plt.show()                                           # not shown
```
We iterate by rounds of *m* iterations. Each round is an *epoch*. So this ^ code goes through the training set 50x.

To preform Linear Regression using SGD with Scikit-Learn, use the SGDRegressor class which defaults to optimizing the squarred error cost function. The following runs 50 epochs, starting with a learning rate of eta=0.1, using the default learning schedule and doesn't use any regularization. 
```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
``` 
--> gives you an intercept and coef very cose to one returned by Normal equation.

### Mini-Batch Gradient Descent
Just computes the gradients on small random sets of instances called *mini-batches*. 

The algo's progress in a param space is less erratic than with stochastic GD, esp with fairly large mini-batches, so this method will get closer to global minimum. But it is harder to esape local minima

## Polynomial Regression
For when you have non linear data but still can fit it using a linear model... Just add powers of each feature as new features, then train a linear model on this extended set of features.

