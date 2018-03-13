# ML

## **Supervised vs [unsupervised]**
Input: examples, in terms of features (like in chess feature would be what’s in each square of board. For medical diagnosis it might be results of the test. For images it might be red blue green values)
Output: 
Classification: you have to answer which kind of animal, do they have the disease or not
Regression: output is a number (like how much money we expect to make)

### Unsupervised
Look for patterns
Bunch of related subproblems:
	Clustering (how is the data clustered?)
	Dimensionality reduction (literally reduce dimensions, what are the most important dimensions that capture the data set?)
	Anomaly detection
	Association rule learning (have all your data, find what’s similar– what do people tend to buy together)

### Reinforcement:
Here’s the question, what’s your answer. It’s either right or wrong. 

### Semi-supervised learning:
Gives examples and has correct answers for some of them

#### *How are you going to train it?*
Online:
	Feed examples one at a time
Batch training:

#### *What type of model?*
Instance based: Keeps around all of the training data. Ex: nearest neighbor approach. You have a new ex you want to classify. Find training instance closest to that and assume it’s in the same category
Model based: have formula/decision tree or something that encapsulates what you’ve learned so you don’t have to keep training data around
	Cost function: how many of these training examples did I get wrong? What’s the average distance between my guess adn the correct guess. Now adjust model to minimize that cost function

## Book set up:
Set up code not printed in book: available on github for website. How to run as a jupitor notebook?
Import os – for file processing?
np.random.seed(42) – always get same sequence of random. (everytime they run program it will be same) 
%matplotlib inline – put it inline with the code

SciPy is a forum where people complain about bugs

#### Code ex 1-1
[Getting data into a useful format for us]
Produces plot, gives a new country, Cyprus, to guess. 
Np.c_ does something to get data into right shape
train the model –  
Make a prediction for Cyprus… let’s make a new x matrix

### Idea of model (neural network, decision tree, etc.)
Use library (like tensorflow)
Or understand math and implement it yourself

### Overfitting solutions: Model does well on training data, but not on data
- Use a simpler model
- Fewer degrees of freedom
- Regularize (put a penalty on an overly fancy model. Add it into your cost function)
- Get more data

### Underfitting: Model is so simple doesn’t do well even on training data
- Use a more complex model
- Get better features

*Data snooping* (looking at testing data before last minute and making decisions based on it.

Use three sets, training, test on validation, final one for test (which you haven’t seen before). The validation one is where you choose the one.
