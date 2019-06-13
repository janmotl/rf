from numpy import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from stratified_choice import stratified_choice


class RFChallenger:
    """
    Online Random Forest with weighted trees.
    The feature selection probability is provided by matlab. In short,
    the new feature is always selected. While the old features are sampled
    to maintain constant mtry.
    The tree weights are adjusted by matlab as well.
    """

    # Static matlab engine (one instance is enough)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(r'.', nargout=0)

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

    def addFeature(self, x):
        """ We treat all features as numerical. Since we are interested only in contrasting challenger vs. baseline, it is ok. """

        # Append column
        self.X = column_stack((self.X, x))
        ncol = size(self.X_t, 1)
        col = size(self.X, 1)

        # Unweighted feature use count
        feature_use_count = zeros((1, ncol))

        # Sample old features
        if col > 1:
            old_features_pack = array(RFChallenger.eng.choiceMtry(col - 1, self.mtry, self.n_jobs), dtype=bool)
        else:
            old_features_pack = ones((self.n_jobs, 0), dtype=bool)

        # Complement the old features with the new feature
        features = column_stack((old_features_pack, ones((self.n_jobs, 1), dtype=bool)))

        # Update the use count
        feature_use_count[0, 0:col] += sum(features, axis=0)

        # Get tree weight
        if col > 1:
            # Note: Used in the challenger. In the baseline, it may result into division by zero
            if feature_use_count[0, col-1] == mean(feature_use_count[0:col-1]):
                tree_weight = 1
            else:
                tree_weight = mean(self.virtual_usage_counts[0:col-1]) / (feature_use_count[0, col-1] - mean(feature_use_count[0:col-1]))
            self.virtual_usage_counts = self.virtual_usage_counts + tree_weight*feature_use_count
        else:
            tree_weight = 1
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
            self.predictions = column_stack((self.predictions, tree_weight * tree.predict_proba(self.X_t[:, flatnonzero(features[i,:])])[:, 1]))

    def getAUC(self):
        prediction = nanmean(self.predictions, axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_t, prediction, pos_label=1)
        return metrics.auc(fpr, tpr)
