---
layout: post
title: "Data Science Interview"
date: 2020-03-23 15:00:00 +0100
---

- https://twitter.com/Al_Grigor/status/1230818076578459649
- https://hackernoon.com/160-data-science-interview-questions-415s3y2a
- https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md


## Supervised machine learning

**What is supervised machine learning? 👶**

Supervised machine learning is a method where the prediction and performance of a model is *supervised* by the true prediction of the training data. Each input (features) also comes with a true output (target). We categorize between 
- Regression: The output of the model is continuous, for example in [Linear Regression]()
- Classification: The output of the model is discrete, for example in [Logistic Regression]()

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

In regression, the prediction are continous values. There exists several models for regression problems:
- Linear Regression, including regressions using polynomials or other functions, regularization mehtods like Ridge Regression (L2), Lasso Regression (L1, feature selection)
- Decision Tree Regression
- Regression using Neural Networks

**What is linear regression? When do we use it? 👶**

Linear Regression is a modeling for regression problems. We use it when there exists a linear dependence between the (modified) features and its targets. In specific, in linear regression we solve following linear optimization problem

$$
\min_{b} \Vert \phi(X)b - y \Vert^2
$$

where $$\phi(X)$$ is <!-- TODO -->

**What’s the normal distribution? Why do we care about it? 👶**

$$
f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( -\frac{1}{2} \left( \frac{x-\mu}{\sigma} \right)^2  \right)
$$

**How do we check if a variable follows the normal distribution? ‍⭐**

- histogram
- chi-square-test? TODO

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

- TODO
- don't use normal but log normal, see https://www.investopedia.com/articles/investing/102014/lognormal-and-normal-distribution.asp
- use the right data for the use case, prices for luxury houses come from a different distribution than prices for most "common" houses

**What are the methods for solving linear regression do you know? ‍⭐️**

- TODO: penrose pseudo inverse, Gaussian elimination, LU decomposition, etc...
- DON'T use gradient descent or other inexact solvers if we can solve it 

**What is gradient descent? How does it work? ‍⭐️**

$$
x_{t+1} = x_t - \alpha_t G(x_t)
$$

where TODO alpha conditions

**What is the normal equation? ‍⭐️**

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

Approximation of the *true* gradient. Instead of using the full training data, we use a small fraction. Typical sizes, 32 or 64.

**Which metrics for evaluating regression models do you know? 👶**

**What are MSE and RMSE? 👶**

- Mean Square Error and Root Mean Square Error

## Validation

**What is overfitting? 👶**

Our goal is to model the true distribution of a given dataset. However, any dataset is only an approximation of that distribution. Because the optimization is based on this approximation only, at some point, we optimize our model too far such that it works very well with the approximation but not very well with the *true* distribution. Therefore, the predictions do not work well with new data. The model focuses on properties that are only present in the approximation but not in the *true* distribution.

**How to validate your models? 👶**

Measurable: Observe loss functions / objective function, metrics like accuracy, precision, etc.
Subjective: evaluate output (e.g. in image generation), conduct user studies

**Why do we need to split our data into three parts: train, validation, and test? 👶**

We need to split our data because we do not want to overfit out model. It is possible to overfit it by the model parameters or by the choosen hyperparameters. Therefore, we need a training *and* validation dataset to be resilient against both. We use the test set to evaluate the final performance of the model choice, which is (more or less) independent of the training and validation data.

Smaller datasets need more validation and test data to evaluate the learning, for example: 60% training, 20% validation, 20% test.
Bigger datasets need less data for evaluation, e.g. 80% training, 10% validation, 10% test.

**Can you explain how cross-validation works? 👶**

1. Split data into training and test data
2. Split training data into training and validation data
3. Optimize over training data and use validation data to find the best hyperparameters.

Techniques:
- $$k$$-fold cross-validation
- Leave one out cross-validation, i.e. $$k = n$$
- Stratified cross-validation to make sure that each subset is balanced
- Cross-validation for time series data

**What is K-fold cross-validation? 👶**

Problem: How do we know that our performance is not mainly attributed due to the validation data?
Solution: For each set of hyperparameters, perform $$k$$ runs and use a different subset of $$n / k$$ data samples to evaluate the performance for given hyperparameters. Then, aggregate the results.

<!-- TODO image -->
 
**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

see https://stats.stackexchange.com/questions/27730/choice-of-k-in-k-fold-cross-validation

## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Prediction onto a discrete output space. There are several options that depend on the nature of the problem and the available hardware for training and prediction:
- Logistic Regression
- k-Nearest-Neighbors
- Decision Trees and Random Forrests
- Support Vector Machines
- Neural Networks

**What is logistic regression? When do we need to use it? 👶**

Logistic Regression is a classification method for supervised learning. It assumes that there is a linear dependency between the features and the prediction, i.e. the outcome depends on the sum of the inputs (and the model's weights). 

We optimize over following loss function (without regularization), which itself is non-linear. That means, we cannot solve it directly as we did in Linear Regression but need dedicated solvers such as SGD.

$$
\min_b \Vert \sigma(Xb) - y \Vert^2
$$

**Is logistic regression a linear model? Why? 👶**

Yes, because the features itself can only model linear dependencies. Nevertheless, the optimization problem itself is non-linear.

**What is sigmoid? What does it do? 👶**

The sigmoid function is defined as

$$
\phi(z) = \frac{1}{1 + e^{-z}}
$$

It maps values from $$(-\infty, \infty)$$ onto $$(0, 1)$$ to give them a probabilistic interpretation. We use the sigmoid function when we have a binary classification problem. When we have more than 2 labels, we use the softmax function which is a generalization of the sigmoid function. It is defined as

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

**How do we evaluate classification models? 👶**
**What is accuracy? 👶**
**What is the confusion table? What are the cells in this table? 👶**
**What is precision, recall, and F1-score? 👶**

Derive the confusion matrix and related metrics. Then, compare them and use the model that is the most appropiate for the use case. For example, for some classifications we prioritize a low "False Negative Rate" over a low "False Positive Rate".

Example based on [this](https://www.youtube.com/watch?v=Kdsp6soqA7o) video

||**has heart disease**|**no heart disease**||
|**predict "heart disease"**|True Positives|False Positives|Precision, Positive Prediction Value|
|**predict "no heart disease"**|False Negatives|True Negatives||
||True Positive Rate, Recall, Sensitivity |False Positive Rate, Fall-out|F1-Score|
|||True Negative Rate, Specificity, Selectivity||

- *"true"*: prediction coincides with reality
- *"false"*: prediction does not coincides with reality
- *"positives"*: prediction of heart disease, regardless of reality
- *"negatives"*: prediction of no heart disease, regardless of reality

**Positive Prediction Value, Precision**: How many selected items are relevant?

- proportion of positive predictions, that were correctly classified. Can be very useful, if data is imbalanced and there are many Negatives (as it is the case with rare disease)
- Search Engine Example: "how useful the search results are"

$$
\mathrm{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}
$$

**True Positive Rate, Recall, Sensitivity**: How many relevant items are selected?

- "percentage of patients *with* heart disease that were correctly identified". Useful metric if we care about positive cases.
- recall is "how complete the results are".


$$
\mathrm{TPR} = \mathrm{recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

**False Positive Rate, Fall-out**: Probability of a false alarm, i.e. false positives

$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
$$
 
**Specificity**: "percentage of patients *without* heart disease that were correctly identified. Useful metric if we care about negative cases.

$$
\mathrm{specificity} = \frac{\mathrm{TN}}{\mathrm{FP}+\mathrm{TN}}
$$

**Accuracy**: Ratio between correctly classified samples and total sample size

$$
\mathrm{accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}} \in [0, 1]
$$

**Balanced Accuracy**: Average between TPR and TNR which is useful if data is unbalanced

$$
\text{balanced accuracy} = \frac{\mathrm{TPR} + \mathrm{TNR}}{2}
$$

**F1-Scores**: Harmonic mean between precision and recall

$$
\mathrm{F}_1 = 2 \cdot \frac{\mathrm{PPV} \cdot \mathrm{TPR}}{\mathrm{PPV} + \mathrm{TPR}} \in [0, 1]
$$

**Matthews correlation coefficient**: Useful evaluation metric, even if data is unbalanced. Value of $$0$$ means random prediction.

$$
\mathrm{MCC} = \frac{\mathrm{TP} \cdot \mathrm{TN} - \mathrm{FP} \cdot \mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}} \in [-1, 1]
$$

**Is accuracy always a good metric? 👶**

It is not very useful if data is unbalanced, i.e. the sizes of the different categories vary.

**Precision-recall trade-off ‍⭐️**
**What is the PR (precision-recall) curve? ‍⭐️**

- measures the usefulness and the completeness of the classification.
- $$\mathrm{recall} \approx 0$$ implies $$\mathrm{precision} \approx 1$$
- $$\mathrm{recall} \approx 1$$ implies $$\mathrm{precision} \approx 0$$

TODO

**What is the ROC curve? When to use it? ‍⭐️**

- "Receiver Operating Characteristic"
- Decreasing the threshold (e.g. for a logisitc regression) increases the TPR, because we correctly more True Positives (and less False Negatives).
- However, this also increases the FPR, because we get more often a false alarm, i.e. higher False Positives (and less true negatives)
- ROC visualizes this tradeoff between the TPR (recall) and the FPR (fall-out, false alarm) where we map the threshold to different TPR and FPR values. This allows us to summarize many confusion matrices in one graph. It also shows, which threshold is better than others.
- $$\mathrm{TPR} \approx 0$$ implies $$\mathrm{FPR} \approx 0$$ and $$\mathrm{TPR} \approx 1$$ implies $$\mathrm{FPR} \approx 1$$.
- This can be seen by the following idea: assume you have a classifier that *always* returns true. Then, it correctly identifies the Positives in *every* case. However, it also predicts that the Negatives are positives in *every* case, resulting in a 100% false-alarm.

**What is AUC (AU ROC)? When to use it? ‍⭐️**

AUC is the area under the ROC. It allows us to compare different classification models without defining a specific threshold. Models with higher AUC might perform better than the ones with lower AUC and might be more suitable for more use-cases.

**What is the area under the PR curve? Is it a useful metric? ‍⭐️**

The AUC is the integral of the ROC. The higher the AUC, the better is the trade-off between TPR and FPR. In general, we favor models with higher ROC over models with lower ROC.

**How to interpret the AU ROC score? ‍⭐️**

TODO

**In which cases AU PR is better than AU ROC? ‍⭐️**

When the data is imbalanced because Precision is not affected by it.

**What do we do with categorical variables? ‍⭐️**
**Why do we need one-hot encoding? ‍⭐️**

TOOD, one hot encoding, TODO why

## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

$$X$$ will not be fully ranked, TODO.


**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

1. Inverting $$X^T X$$ becomes numerically difficult or inaccurate.
2. Small changes in the model parameters can result in large changes in the prediction
2. Interpreting the model's parameters becomes undefined, because we cannot change one feature only while fixing all other paramters.

**What is regularization? Why do we need it? 👶**

- We prefer "simpler" models over more "complicated" ones, i.e. models that are stable, don't require many parameters and use small values only. That way, noise and numerical errors won't have a large effect.
- we reduce the variance of the model parameter by adding bias to our estimation though.

**Which regularization techniques do you know? ‍⭐️**

- L1 regularization, Lasso regression
- L2 regularization, Ridge regression
- Dropout for neural networks

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

L1 and L2 regularization

**How does L2 regularization look like in a linear model? ‍⭐️**

We optimize over following problem

$$
\begin{aligned}
\hat b &= \arg\min_{b} \frac{1}{2} \Vert Xb - y \Vert^2 + \frac{\lambda}{2} \Vert b \Vert^2 \\
&= (X^T X + \lambda I)^{-1} X^T y
\end{aligned}
$$

Note that we acutally solve $$(X^T X + \lambda I) \hat b = X^T y$$ in practice using Gauss-elimination, LU decomposition, etc.

**How do we select the right regularization parameters? 👶**

We treat it as a hyper-parameter and use know methods to find the best one. For example, grid search in combination with cross-validation.

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

L2 regularization favors "small" model weights because the L2 norm is a "ball" 

**How L1 regularization looks like in a linear model? ‍⭐️**

We optimize over following problem

$$
\hat b = \arg\min_{b} \frac{1}{2} \Vert Xb - y \Vert^2 + \frac{\lambda}{2} \Vert b \Vert
$$

Solving this problem is more difficult because it does not have an analytical solution. In addition, it is not even differentiable. However, the L1 norm is proximable and we can utilize Forward-Backward splitting or ADMM to converge to a solution.

**What’s the difference between L2 and L1 regularization? ‍⭐️**

- L1: sparse solution
- L2: small valued solution

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Yes, this is also known as L1,2 regularization and is sometimes useful in computer vision tasks for example

TODO: formula

**What’s the interpretation of the bias term in linear models? ‍⭐️**

**How do we interpret weights in linear models? ‍⭐️**

The effect of one unit change in the variable on the prediction while fixing all other parameters.

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

A higher weight has a higher contribution to the prediction. However, both weights could be important for the prediction.

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

L1 when not much data is available.

TODO

## Feature selection

TODO 

**What is feature selection? Why do we need it? 👶**

- Dimensionality reduction
- shrinkage of data
- removing linear dependent features in linear models

**Is feature selection important for linear models? ‍⭐️**

- Yes, if some features linear dependent, then we run into problems. TODO which problems?

**Which feature selection techniques do you know? ‍⭐️**

- L1 Regularization
- PCA
- Greedy elimination: TODO
- Recursive feature elimination: TODO
- TODO metrics like ... (see bachelor thesis)

**Can we use L1 regularization for feature selection? ‍⭐️**

Yes, because the "edges" of a L1 norm lie on the axis, L1 regularization favors sparse solutions where some components are 0.

**Can we use L2 regularization for feature selection? ‍⭐️**

No, L2 regularization favours smaller values over larger one, but does not make it sparse.

## Decision trees
What are the decision trees? 👶
How do we train decision trees? ‍⭐️
What are the main parameters of the decision tree model? 👶
How do we handle categorical variables in decision trees? ‍⭐️
What are the benefits of a single decision tree compared to more complex models? ‍⭐️
How can we know which features are more important for the decision tree model? ‍⭐️

## Random forest
What is random forest? 👶
Why do we need randomization in random forest? ‍⭐️
What are the main parameters of the random forest model? ‍⭐️
How do we select the depth of the trees in random forest? ‍⭐️
How do we know how many trees we need in random forest? ‍⭐️
Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️
What are the potential problems with many large trees? ‍⭐️
What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀
What happens when we have correlated features in our data? ‍⭐️

## Gradient boosting
What is gradient boosting trees? ‍⭐️
What’s the difference between random forest and gradient boosting? ‍⭐️
Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️
Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️
Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀
What are the main parameters in the gradient boosting model? ‍⭐️
How do you approach tuning parameters in XGBoost or LightGBM? 🚀
How do you select the number of trees in the gradient boosting model? ‍⭐️

## Parameter tuning
Which parameter tuning strategies (in general) do you know? ‍⭐️
What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️

## Neural networks

**What kind of problems neural nets can solve? 👶**

A neural network can (theoretically) approximate any function. It is widely used in

- Regression
- Classification
- Dimensionality Reduction/Compression, Clustering, e.g. via (Variational) Autoencoders
- Generative Models, e.g. via Generative Adversarial Networks
- Reinforcement Learning, e.g. (Deep) Q-Learning

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

It consists of an input layer, hidden layer and an output layer, where a hidden layer consists of

- linear component
- non-linear activation function, for example $$\text{ReLU}$$

$$
f(x) = \sigma_n( \ldots \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2) \ldots + b_n)
$$

**Why do we need activation functions? 👶**

We need to introduce non-linearity. Otherwise, our neural network function can be simplified into a simple linear model.

**What are the problems with sigmoid as an activation function? ‍⭐️**

Sigmoid is defined as $$\sigma(x) = \frac{1}{1 + \exp(-x)}$$ and it's range is $$(0, 1)$$. Problem: vanishing gradients since its output does not exceed 1. So each layer, we multiply values smaller than 1 together which results in even smaller values.


**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

ReLU is another activation function whose range is $$[0, \infty)$$ in contrast to the range of the sigmoid $$(0, 1)$$ or tanh $$(-1, 1)$$. That means, gradients can grow and we can counteract the vanishing gradients problem. Even though it is not differentiable at $$x = 0$$, in practice we can still use it. 

$$\mathrm{ReLU}(x) = \max{0, x}$$

**How we can initialize the weights of a neural network? ‍⭐️**

- randomly (uniform): potential problem of vanishing / exploding gradients
- Xavier’s random weight initialization: solves this problem

For ReLU: Sample weights from $$\mathcal{N}(0, 1)$$ and multiply it with $$\sqrt{\frac{2}{m}}}$$ where $$m_i$$ is the number of inputs for its layer $$i$$

**What if we set all the weights of a neural network to 0? ‍⭐️**

All neurons will calculate the same value (at the beginning, during training and also at the end) and therefore cannot capture the distribution of the training data.

**What regularization techniques for neural nets do you know? ‍⭐️**
**What is dropout? Why is it useful? How does it work? ‍⭐️**

- Dropout: Randomly "deactivate" neurons with a certain probability. This will help the network to distribute information across several neurons (instead of just a few). 

TODO: write more about dropout

## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

Because the optimization problem for neural networks is very difficult, we cannot find an analytical soltuion to it. Therefore, we need iterative methods to find a (local) optimium instead. Many of these mehtods (SGD, ADAM, ...) require that the neural network is differentiable and that we can calculate the gradient of it. To calculate the gradient, we use the Backpropagation algorithm, which works very well, as long as all parts of the neural network are differentiable (or nearly differentiable, see ReLU).

**Which optimization techniques for training neural nets do you know? ‍⭐️**

- GD and SGD
- Rprop
- RMS Prop, ADAM

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

1. Specify batch size $$M$$
2. Specify learning rate $$\alpha_k$$ for iteration $$k$$
3. Aggregate gradients $$G_{k,m}$$ of our neural network using $$M$$ training samples only (instead of the full batch)
4. Update optimization variable

$$
w_{k+1} = w_k - \alpha_k \sum_{m=1}^M G(w_; \{x_m, y_m\})
$$

**What’s the learning rate? 👶**
What happens when the learning rate is too large? Too small? 👶

also known as step size. It defines "how big" of an optimization step we want to take. A large learning rate can accelerate the learning but we might miss certain optima. On the other hand, a smaller learning rate slows down the training.

**How to set the learning rate? ‍⭐️**

- Fixed learning rate: We treat it as a hyperparameter
- Adaptive learning rate: We use an algorithm (such as RMS Prop, ADAM, Hypergradient-Descent) to *adaptively* change the learning rate for appropiate values 

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

TODO: write down formula and advantages, see lecture

**When would you use Adam and when SGD? ‍⭐️**

I would always prefer ADAM over SGD in most application. TODO: is there actually a (non-academic) reason?

**Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️**

We *want* a constant learning rate because we do not want to slow down our training. However, in practice our algorithms on the given problems often overshoot optima. Therefore, for SGD, ADAM and similar, we rather want to use an adaptive learning rate that decreases during training. 

**How do we decide when to stop training a neural net? 👶**

At the beginning of the training, the training and the validation error decrease. But, at some point, the validation error increases again while the training error continues to decrease. We want to stop the training once we confirmed that the validation error rises again. Otherwise, we overfit the model.

**What is model checkpointing? ‍⭐️**

Save weights of the model during training to be prepared against computation issues (loss of power, ...)

**Can you tell us how you approach the model training process? ‍⭐️**

???

## Neural networks for computer vision
How we can use neural nets for computer vision? ‍⭐️
What’s a convolutional layer? ‍⭐️
Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️
What’s pooling in CNN? Why do we need it? ‍⭐️
How does max pooling work? Are there other pooling techniques? ‍⭐️
Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀
What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️
What kind of CNN architectures for classification do you know? 🚀
What is transfer learning? How does it work? ‍⭐️
What is object detection? Do you know any architectures for that? 🚀
What is object segmentation? Do you know any architectures for that? 🚀

## Text classification
How can we use machine learning for text classification? ‍⭐️
What is bag of words? How we can use it for text classification? ‍⭐️
What are the advantages and disadvantages of bag of words? ‍⭐️
What are N-grams? How can we use them? ‍⭐️
How large should be N for our bag of words when using N-grams? ‍⭐️
What is TF-IDF? How is it useful for text classification? ‍⭐️
Which model would you use for text classification with bag of words features? ‍⭐️
Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️
What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️
Do you know any other ways to get word embeddings? 🚀
If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️
Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️
How can you use neural nets for text classification? 🚀
How can we use CNN for text classification? 🚀

## Clustering
What is unsupervised learning? 👶
What is clustering? When do we need it? 👶
Do you know how K-means works? ‍⭐️
How to select K for K-means? ‍⭐️
What are the other clustering algorithms do you know? ‍⭐️
Do you know how DBScan works? ‍⭐️
When would you choose K-means and when DBScan? ‍⭐️

## Dimensionality reduction
What is the curse of dimensionality? Why do we care about it? ‍⭐️
Do you know any dimensionality reduction techniques? ‍⭐️
What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️

## Ranking and search
What is the ranking problem? Which models can you use to solve them? ‍⭐️
What are good unsupervised baselines for text information retrieval? ‍⭐️
How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️
What is precision and recall at k? ‍⭐️
What is mean average precision at k? ‍⭐️
How can we use machine learning for search? ‍⭐️
How can we get training data for our ranking algorithms? ‍⭐️
Can we formulate the search problem as a classification problem? How? ‍⭐️
How can we use clicks data as the training data for ranking algorithms? 🚀
Do you know how to use gradient boosting trees for ranking? 🚀
How do you do an online evaluation of a new ranking algorithm? ‍⭐️

## Recommender systems
What is a recommender system? 👶
What are good baselines when building a recommender system? ‍⭐️
What is collaborative filtering? ‍⭐️
How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️
What is the cold start problem? ‍⭐️
Possible approaches to solving the cold start problem? ‍🚀

## Time series
What is a time series? 👶
How is time series different from the usual regression problem? 👶
Which models do you know for solving time series problems? ‍⭐️
If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️
You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️
You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️
What are the problems with using trees for solving time series problems? ‍⭐️
