# Types of ML
### Types of Learning: Supervised?
| Type | Description|Examples |
| --- | --- | --- |
| **Supervised learning** | Training data you feed to the algorithm includes desired labels (solutions) | Ex. classification– spam filter classifying things as spam or not  Ex. predicting a target numeric value give a set of features.
| **Unsupervised Learning** | Training data is unlabeled
| **Semi supervised learning** | Partially labeled data, but largely unlabeled
| **Reinforcement learning** | The agent (learning system) observes environment and preforms actions and gets rewards/penalties in return

##### Unsupervised learning– Common Algorithms
- *clustering*– detecting similar groups
- *hierarchical clustering*– same as clustering but with groups having subdivisions
- *visualization*– feed in lots of unlabeled data, they output a 2D or 3D representation that can be plotted
##### Unsupervised learning tasks
- *Anomaly detection*–  detecting anomalies
- *Dimensionality reduction*– goal is to simplify data without losing too much information
- *Association rule learning*– using lots of data to find relations between attributes. Like a supermarket learning people buy hotdogs and beer together 

### Types of Learning: Batch vs. Online
**Batch:** System cannot learn incrementally. Must be trained using all available data. Once trained it is launched into production and does not continue to learn.
(also called offline learning)
For the system to know about new data you train a whole new version from scratch with the full dataset.

**Online:** Can train system incrementally with individual data instances or mini-batches.
Once an online learning system has learned new instances, you don’t need to store them so it’s good for space. 

### Tpes of Learning: Instance vs Model-Based
**Instance-based:** System learns examples by heart and then generalizes to new cases based on a similarity to this one so it must also be a spam email

**Model-based:** Build model from example and then make predicutions based off of the model. 
Like comparing better life index with a country's GDP per capita


#### OVERFITTING DATA: 
Overgeneralizing data. Like taking it too seriously (like all girls in LA are blonde because the first two you met were). 
This model will perform well on training data, but does not generalize well 

To reduce risk of this you can constrain a module to make it simpler (regularization). 
A model has 2 degrees of freedom if it has two features, but if you set one of those to a constant (or a range), it only has one. 
Or you can force it to keep it small so you have between 1-2 degrees of freedom so it’s simpler but more complex than a model with just 1 degree.
#### UNDERFITTING DATA:
Model is too simple to learn underlying structure of data
*No free lunch theorem:* If you make no assumptions about the data, there is no reason to prefer one model over another. 
For some datasets a linear model would work best, for others a neural network. The only way to know which model is the best is to evaluate them all. In practice this isn’t really possible so you have to make some assumptions to narrow things down a level.
