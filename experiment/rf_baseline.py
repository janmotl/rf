from numpy import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from stratified_choice import stratified_choice


class RFBaseline:
    """
    Online Random Forest with unit tree weight.
    """

    def __init__(self, y, X_t, y_t, n_jobs, mtry, random_state):
        self.X = empty((len(y), 0))
        self.y = y
        self.X_t = X_t
        self.y_t = y_t
        self.n_jobs = n_jobs
        self.mtry = mtry
        self.random_state = check_random_state(random_state)

        self.virtual_tree_count = 0
        self.virtual_usage_counts = zeros(size(self.X_t, 1))
        self.tree_weights = zeros(n_jobs * size(self.X_t, 1))
        self.predictions = empty((size(self.X_t, 0), 0))

    def fit(self, x):
        """ We treat all features as numerical. Since we are interested only in contrasting challenger vs. baseline, it is ok. """

        # Append column
        self.X = column_stack((self.X, x))
        ncol = size(self.X_t, 1)
        col = size(self.X, 1)

        # Unweighted feature use count
        feature_use_count = zeros((1, ncol))

        # BASELINE: There has to be at least one feature for the tree. And that's all we require. Can be slow.
        while True:
            if col>1:
                p = self.get_mtry(col)
            else:
                p = self.mtry
            features = random.choice([False, True], (self.n_jobs, col), p=[1-p, p])
            if all(sum(features, 1)>0):
                break

        # Update the use count
        feature_use_count[0, 0:col] += sum(features, axis=0)

        # BASELINE: Update virtual_usage_counts for QC
        if col > 1:
            self.virtual_usage_counts = self.virtual_usage_counts + feature_use_count
        else:
            self.virtual_usage_counts = feature_use_count

        # Debug
        print(self.virtual_usage_counts, flush=True)

        for i in range(self.n_jobs):
            # Sample training instances
            samples = stratified_choice(self.y, replace=True)

            # Train the tree
            tree = DecisionTreeClassifier(random_state=self.random_state, max_depth=6)
            tree.fit(self.X[ix_(samples, features[i,:])], self.y[samples])

            # Score testing instances
            self.predictions = column_stack((self.predictions, tree.predict_proba(self.X_t[:, flatnonzero(features[i,:])])[:, 1]))

    def get_mtry(self, ncol):
        # We want to preserve the constant mtry regardless of the count of the features (we have to take into account the
        # new feature is always present while the rest is sampled).
        # This is important for a fair comparison.
        return (self.mtry*ncol-1) / (ncol-1)

    def get_auc(self):
        prediction = nanmean(self.predictions, axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_t, prediction, pos_label=1)
        return metrics.auc(fpr, tpr)
