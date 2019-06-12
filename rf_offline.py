from numpy import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


def offline(X, y, X_t, y_t, n_jobs, mtry, seed):
    predictions = empty((size(X_t, 0), 0))

    while True:
        features = random.rand(n_jobs, size(X, 1)) < mtry
        if all(sum(features, 1) > 0):
            break

    for i in range(n_jobs):
        # Train the tree
        tree = DecisionTreeClassifier(random_state=seed, max_depth=6)
        samples = random.choice(size(X, 0), size(X, 0), replace=True)
        tree.fit(X[ix_(samples, features[i, :])], y[samples])

        # Score the testing samples (it may return just a single column...)
        try:
            predictions = column_stack((predictions, tree.predict_proba(X_t[:, flatnonzero(features[i, :])])[:, 1]))
        except Exception:
            print('Damn it')

    prediction = nanmean(predictions, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(y_t, prediction, pos_label=1)
    auc_offline = metrics.auc(fpr, tpr)

    return auc_offline