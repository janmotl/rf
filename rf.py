from numpy import *
from sklearn.tree import DecisionTreeClassifier

from stratified_choice import stratified_choice


class RF:
    """
    Random Forest on a stream of features.
    We may add a single or multiple features at once.
    We may add a single or multiple trees at once.
    The used decision tree is from sklearn -> see the documentation for sklearn.tree.
    """

    def __init__(self, y, n_jobs=10, mtry=0.1, seed=2001):
        """
        Initialize the random forest.

        :param y: the label (a vector with the class labels)
        :param n_jobs: how many new trees to train at each update (integer in range 1..inf)
        :param mtry: the feature sampling probability (double in range 0..1)
        :param seed: for reproducibility
        """
        self.y = y
        self.n_jobs = n_jobs
        self.mtry = mtry
        self.seed = seed

        self.X = empty((len(y), 0))
        self.feature_usage_count = array([], int)
        self.trees = [] # Contains tuples: (tree, feature_set, weight)

    def fit(self, X):
        """
        We treat all features as numerical. Since we are interested only in contrasting challenger vs. baseline, it is ok.

        :param X: vector or a matrix with the new feature(s)
        """

        n_old_cols = size(self.X, 1)
        if len(shape(X)) == 1:
            n_new_cols = 1
        else:
            n_new_cols = size(X, 1)
        random.seed(self.seed)

        # Append column(s)
        self.X = column_stack((self.X, X))

        # Sample old features (no stratification and no guarantee of the counts - only of the probabilities...)
        mtry = self.get_mtry(n_old_cols, n_new_cols)
        print(mtry, self.mtry)
        old_features_pack = random.choice([False, True], (self.n_jobs, n_old_cols), p=[1 - mtry, mtry])

        # Complement the old features with the new features
        features = column_stack((old_features_pack, ones((self.n_jobs, n_new_cols), dtype=bool)))

        # Get unweighted feature use count of the new trees
        feature_use_count = sum(features, axis=0)

        # Get tree weight and update feature_usage_count
        if n_old_cols > 0 and self.mtry < 1.0:
            tree_weight = mean(self.feature_usage_count[0:n_old_cols]) / (mean(feature_use_count[n_old_cols:n_old_cols+n_new_cols]) - mean(feature_use_count[0:n_old_cols]))
            new_feature_usage_count = tree_weight * feature_use_count
            new_feature_usage_count[0:n_old_cols] += self.feature_usage_count
            self.feature_usage_count = new_feature_usage_count
        else:
            tree_weight = 1
            self.feature_usage_count = feature_use_count

        for i in range(self.n_jobs):
            # Sample training instances
            samples = stratified_choice(self.y, replace=True)

            # Train the tree
            tree = DecisionTreeClassifier(max_depth=6)
            tree.fit(self.X[ix_(samples, features[i, :])], self.y[samples])

            # Store the tree
            self.trees.append((tree, features[i, :], tree_weight))

        return self.feature_usage_count

    def score(self, X_t):
        """
        Returns a weighted average prediction from the individual trees.

        :param X_t: a 2D array to score
        """
        predictions = empty((size(X_t, 0), 0))

        for t in self.trees:
            tree, feature_set, tree_weight = t
            data = X_t[:, 0:len(feature_set)]
            data = data[:, feature_set]
            predictions = column_stack((predictions, tree_weight * tree.predict_proba(data)[:, 1]))

        return mean(predictions, 1)

    def get_mtry(self, n_old_cols, n_new_cols):
        # We want to preserve the constant mtry regardless of the count of the features (we have to take into account the
        # new feature is always present while the rest is sampled).
        # This is important for a fair comparison to normal random forest implementations.
        if n_old_cols == 0:
            return self.mtry    # Not important...
        else:
            return (self.mtry * (n_old_cols + n_new_cols) - n_new_cols) / n_old_cols
