from numpy import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from stratified_choice import stratified_choice


def offline(X, y, X_t, y_t, n_jobs, mtry, seed):
    predictions = empty((size(X_t, 0), 0))

    while True:
        features = random.rand(n_jobs, size(X, 1)) < mtry
        if all(sum(features, 1) > 0):
            break

    for i in range(n_jobs):
        # Sample training instances
        samples = stratified_choice(y, replace=True)

        # Train the tree
        tree = DecisionTreeClassifier(random_state=seed, max_depth=6)
        tree.fit(X[ix_(samples, features[i, :])], y[samples])

        # Score testing instances
        predictions = column_stack((predictions, tree.predict_proba(X_t[:, flatnonzero(features[i, :])])[:, 1]))

    prediction = nanmean(predictions, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(y_t, prediction, pos_label=1)
    auc_offline = metrics.auc(fpr, tpr)

    return auc_offline