from sklearn import metrics

from rf import RF


class RFWrapper:
    """
    Online Random Forest.
    """

    def __init__(self, y, X_t, y_t, n_jobs, mtry, random_state):
        self.rf = RF(y, n_jobs=n_jobs, mtry=mtry, seed=random_state)
        self.X_t = X_t
        self.y_t = y_t

    def fit(self, x):
        self.rf.fit(x)

    def get_auc(self):
        prediction = self.rf.score(self.X_t)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_t, prediction, pos_label=1)
        return metrics.auc(fpr, tpr)
