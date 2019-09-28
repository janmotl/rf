Online Random Forest, which works on a stream of features (and not on a stream of samples).

#### Example

```python
rf = RF(y)                          # Initialize a new random forest with label y
rf.fit(X1)                          # Train new trees based on features [X1] 
prediction = rf.score(X1_t)         # Score data based on features X1
rf.fit(X2)                          # Train new trees based on features [X1, X2]
prediction = rf.score(X1_X2_t)	    # Score data based on features [X1, X2]
```

#### Motivation

Sometimes we want to update a current random forest with a *set of new features*, but retraining the whole model from scratch seems to be wasteful. Can we do better? We can, if we combine the following 3 ideas:

The first idea is to *append the random forest with new trees trained on a random sample of all the features* (old and new). This way we reuse the whole old random forest. The issue with this idea is that the old features end up used in the random forest more frequently than the new features simply because the older features had more opportunities to get selected than the new features. This is bad because we want the trees in the random forest to be as diverse as possible. And based on Shannon's entropy, if we want as diverse trees as possible, the selection rate of all the features (old and new) in the whole random forest must follow a uniform distribution.

The second idea is to make sure that *the new trees use new features more frequently than the old features* and add as many new trees until we get uniform distribution. Unfortunately, this leads to the quadratic growth of the count of the trees with the count of the features. That's inconvenient.

The third idea is to *weight the new trees more than the old trees*. If we weight the new trees right, we can keep the count of the new trees in each update constant. And during the scoring time, instead of calculating the average of the predictions (as in a normal random forest), we calculate the weighted average of the predictions.

#### Algorithm properties

1. Online learning: The count of features can grow. But the count of training samples is fixed.
2. Anytime learning: We do not know the final count of the features. Hence, after each update, the algorithm provides as good ensemble as possible.

#### Implementation

The current implementation is minimalistic (~50 lines of code). Hence the following *implementation* limitations:

1. Only for binary classification 
2. Supports only numerical features
3. Hardcoded scikit tree learner with the default parameters
4. Missing values are unsupported

Each of these limitations can be removed → contributions are welcomed.

#### How to cite

[Learning on a Stream of Features with Random Forest](http://web.ics.upjs.sk/~horvath/itat2019/)  Jan Motl, Pavel Kordík