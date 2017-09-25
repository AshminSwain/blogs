---
title: Linear Model Selection
desc: In a regression setting, the standard linear model the form Y = β<sub>0</sub> + β<sub>1</sub> X<sub>1</sub> + β<sub>2</sub>X<sub>2</sub> + … + β<sub>p</sub>X<sub>p</sub> + ε is commonly used to describe the relationship between a response Y and a set of variable $X_1$, $X_2$ ,.., $X_p$. This linear model framework can be extended to accommodate non-linear relationships. But linear model has distinct advantages in terms of inference and on real-world problems and gives a tough competition to non-linear models. Linear models can be improved by replacing plain least squares fitting with some alternative fitting procedures which can give better prediction accuracy and model interpretability. 
author: ashmin
tags: cross validation k fold bootstrap
image: model_selection.png
layout: post
mathjax: true
permalink: /linear-model-selection/
---

* Do not remove this line (it will not be displayed) 
{:toc}

## Prediction Accuracy

If the relationship between the response and the predictors is approximately linear, the least square estimates would have low bias.

If the number of observations is much larger than the number of variables, then the least squares estimates tend to also have low variance and will perform well on test observations.

But if the number of observations is not much larger than the number of predictors, then there can be a lot of variability in the least squares fit and would result in overfitting and poor predictions on the observations not used to train the model.

If the number of predictor variables are much larger than the number of observations, then the variance is infinite hence the method cannot be used at all. By shrinking the estimated coefficients, the variance can be reduced (resulting in increase in bias). This can be then lead to good improvements in prediction accuracy.

## Model Interpretability

Its often the case that some or many of the predictor variables used in multiple regression model are not associated with the response. Inclusion of such variables makes the model unnecessarily complex. Removing these variables can make the model more interpretable. Feature selection or variable selection should be done for excluding irrelevant variables from a multiple regression model.

There are many alternatives to using least squares to fit. Some of the important methods are:

* Subset Selection: This approach identifies a subset of the predictors that are related to the response. The model is then fit using least squares on the reduced set of variables.

* Shrinkage (Regularization): This approach involves fitting a model involving all the predictors. But the estimated coefficients are shrunken towards zero relative to the least squares estimates. This reduces variance and depending on the type of shrinkage used, some coefficients may be estimated to be exactly zero and hence can be used for variable selection.

* Dimension Reduction: This approach involves projecting the p predictors into a M-dimensional subspace where M < p. This is achieved by computing M different linear combinations or projections of the variables. These M projections are then used as predictors to fit a linear regression model by least squares.

## Subset Selection

### Best subset Selection

For best subset selection, a separate least squares regression is fit for each possible combination of the p predictors. All p models are fit that contains exactly one predictor, all $\begin{pmatrix}p \\ 2 \end{pmatrix} = p(p-1)/2$ models that contains exactly two predictors and so forth. Then all the resulting models are looked upon and the best model is selected.

The problem with best subset selection is that there are $2^p$ possibilities being considered by the best subset selection method.

Following is the algorithm for Best selection model.
* Let  $\mathcal{M}_0$ denote the null model which contains no predictors. This model simply predicts the sample mean for each observation.

* For k = 1,2,…, p:

    a. Fit all $\begin{pmatrix}p \\ k \end{pmatrix}$ models that contains exactly k predictors.

    b. Pick the best among these $\begin{pmatrix}p \\ k \end{pmatrix}$ models and call it  $\mathcal{M}_k$. Here best is defined as having the smallest RSS or the largest $R^2$

* Select a single best model from among  0,…, p using cross-validated prediction error, $C_p$ (AIC), BIC or adjusted $R^2$.

In algorithm, step 2 identifies the best model (on training data) for each subset size in order to reduce the problem from one of the $2^p$ possible models to one to p+1 possible models. Selecting a single best model from the resulting p+1 models must be done with care as RSS of these models decreases monotonically and the $R^2$ increases similarly as the number of features included in the models increase. The model with most features hence would always become the best model but a model with low test error is wished instead a model with low training error. Hence in step 3, cross validated prediction error $C_p$, BIC or adjusted $R^2$ is used in order to select the best model.

This method can be used for other learning methods apart from least squares regression. For e.g. in the case of logistic regression, instead of ordering models by RSS in step 2, another measure called deviance can be used. The smaller the deviance, the better the fit.

Though best subset method is simple and conceptually appealing, it suffers from computational limitations. The number of possible models to be considered increases with p. A problem with 20 predictors can give over a million possible models for the best model to be selected. Techniques like branch and bound can be used to reduce some choices but they fail with large p.

### Stepwise Selection

#### Forward Stepwise Selection

It is a computationally efficient alternative to best subset selection method. While the best subset selection method considers all the $2^p$ possible methods containing subsets of the p predictors, forward stepwise considers a much smaller set of models. The method begins with a model containing no predictors and then adds predictors to the model one at a time, until all of the predictors are in the model. In particular, at each step the variable that gives the greatest additional improvement to the fit is added to the model.

Following is the algorithm for the method:

* Let $\mathcal{M}_0$ denote the null model, which contains no predictors.

* For k = 0,…, p-1:

    a. Consider all p-k models that augment the predictors in $\mathcal{M}_k$ with one additional predictor.

    b. Choose the best among these p-k models and call it $\mathcal{M}_{k+1}$. Here best is defined as having smallest RSS or highest $R^2$.

* Select a single best model from among $\mathcal{M}_0$, …, $\mathcal{M}_p$ using cross-validated prediction error, $C_p$ (AIC), BIC or adjusted $R^2$.

Unlike best subset selection which involves fitting 2p models, forward stepwise selection involves fitting one null model, along with p-k models in the kth iteration for k = 0, …, p-1. This amount for $1 + \sum_{k=0}^{p-1}(p−k) = 1+p(p+1)/2$ models. When p = 20, best subset selection requires fitting 1,048,576 models where as forward selection requires fitting only 211 models which is a substantial difference.

Forward stepwise selection’s computational advantage over best subset selection is clear. Though forward stepwise tends to do well in practice, it is not always guaranteed to find the best possible model out of the $2^p$ models containing subsets of the p predictors. For instance, if in a given dataset there are 3 predictors, the best possible one variable model contains $X_1$ and the best possible two variable model instead contains $X_2$ and $X_3$. Then forward stepwise selection will fail to select the best possible two-variable model because $\mathcal{M}_1$ will contain $X_1$, so $\mathcal{M}_2$ must also contain $X_1$ together with one additional variable.

Forward stepwise selection can be applied even in high dimensional settings where n < p, however in this case, it is possible to construct submodels $\mathcal{M}_0$, ..., $$\mathcal{M}_{n-1}$$ only since each submodel is fit using least squares which will not yield a unique solution if p >= n.

#### Backward Stepwise Selection

This method also provides an efficient alternative to the best subset selection. But unlike forward stepwise selection, it begins with the full least squares model containing all the p predictors, and then iteratively removes the least useful predictor, one at a time.

Following is the algorithm for the method:

* Let $\mathcal{M}_p$ denote the full model containing all the p predictors.

* For k = p, p-1, ..., 1:
    
    a. Consider all k models that contain all but one of the predictors in $\mathcal{M}_k$, for a total of k-1 predictors.
    
    b. Choose the best among these k models and call it  $\mathcal{M}_{k-1}$. Here best is defined as having the smallest RSS or highest $R^2$.

* Select a single best model from among $\mathcal{M}_0$, …, $\mathcal{M}_p$ using cross-validated prediction error, $C_p$ (AIC), BIC or adjusted $R^2$.

Like forward stepwise selection, the backward selection approach searches through only 1+p(p+1)/2 models, and so can be applied in settings where p is too large to apply best subset selection. Also like forward stepwise selection, backward stepwise selection is not guaranteed to yield the best model containing a subset of the p predictors.

Backward selection requires that the number of samples n is larger than the number of variables p, so that the full model can be fit. But forward stepwise can be used even when n < p, so it is the only subset method when p is very large.

#### Hybrid Approaches

The best subset, forward stepwise and backward stepwise selection approaches give similar but not identical models. A hybrid of forward and backward stepwise selection can be used in which variables are added to the model sequentially, but after adding each new variable, the method may remove any variable which no longer provides an improvement in the model fit. This hybrid approach tries to imitate the best subset selection method while retaining the computational advantage of forward and backward stepwise selection.

#### Choosing the Optimal Model

Best subset selection, forward/backward stepwise selection methods result in the creation of a set of models, each of which contain a subset of p predictors. In order to implement these methods, a way to determine which of the models is best is required by knowing the model which gives low test error. The two common approaches to determine test error are:

* Indirectly estimating the test error by making an adjustment to the training error to account for the bias due to overfitting.

* Directly estimating the test error, by using a validation set approach or a cross-validation approach.

### $C_p$, Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC) and Adjusted $R^2$

For a fitted least squares model containing d predictors, the $C_p$ estimate of test MSE is computed using the equation

$C_p = \frac{1}{n}(RSS + 2d\hat{σ}^2)$

where $\hat{σ}^2$ is an estimate of the variance of the error ϵ associated with each response measurement of the standard linear model. The $C_p$ adds a penalty of $2\hat{σ}^2$ to the training RSS in order to adjust for the fact that the training error tends to underestimate the test error. The penalty increases as the number of predictors in the model increases which adjusts the corresponding decrease in training RSS. So when determining which set of models is best, the model with lowest $C_p$ value is selected.

The AIC criterion is defined for a large class of models fit by maximum likelihood. In the case of the standard linear model with Gaussian errors, maximum likelihood and least squares are the same thing. In this case AIC is given by

$AIC = \frac{1}{n\hat{σ}^2}$

For least square models, $C_p$ and AIC are proportional to each other.

BIC is derived from a Bayesian point of view but its similar to $C_p$ and AIC also. For a least square model with d predictors, the BIC is given by 

$BIC = \frac{1}{n}(RSS + log(n)d\hat{σ}^2)$

Like $C_p$, the BIC tend to take on small values for a model with a low test error so models with low BIC values are selected. As log n > 2 for any n > 7, the BIC statistic places a heavier penalty on models with many variables and hence results in selection of models smaller than $C_p$.

Adjusted $R^2$ statistic is another approach for selecting models with many variables. For a least square model with d variables, the adjusted $R^2$ is calculated as 

$Adjusted R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)}$

Unlike $C_p$, AIC and BIC, a large value of $R^2$ indicates a model with a small test error. Maximizing adjusted $R^2$ means minimizing RSS/(n-d-1). While RSS always decreases as the number of variables in the model increases, RSS/(n-d-1) may increase or decrease due to the presence of d in the denominator.

In the case of adjusted $R^2$, once all the correct variables have been included in the model, adding additional noise variables will lead to very small decrease in RSS. So theoretically, the model with highest $R^2$ will have only correct variables and no noise variables.

### Validation and Cross Validation

As an alternative to the methods mentioned earlier, the test error can be directly estimated using the validation set and cross-validation methods. The validation set error or the cross validation error for each model under consideration can be computed and then the model for which the resulting estimated test error is smallest is selected. In comparison to $C_p$, AIC, BIC and adjusted $R^2$, these methods provide a direct estimate of the test error and it makes fewer assumptions about the true underlying model.

## References

- James G., Witten D., Hastie T., Tibshirani R. (2013). An introduction to Statistical Learning. New York, NY: Springer