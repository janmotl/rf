import unittest

from numpy.testing import assert_almost_equal

from rf import RF
from numpy import *

# Data
y = array([0, 1, 0, 1, 1])
X0 = array([[0], [1], [0], [1], [1]])
X1 = array([[0, 1], [1, 0], [0, 1], [1, 2], [1, 3]])
X2 = array([[0, 3], [1, 2], [0, 0], [1, 2], [1, 3]])

# Numpy warnings to errors
seterr(all='raise')

class TestWeight(unittest.TestCase):

    def test_incremental_learning(self):
        # Test that we can initialize the RF, add a feature, score, add features, score.

        rf = RF(y, mtry=0.8, n_jobs=2, seed=2001)
        rf.fit(X0)
        prediction1 = rf.score(X0)
        rf.fit(X1)
        prediction2 = rf.score(column_stack((X0, X1)))

        assert_almost_equal(prediction1, y, err_msg="Feature X0 is a leaking feature - overfit on it!")
        assert_almost_equal(prediction2, y, err_msg="Feature X0 is a leaking feature - overfit on it!")

    def test_vector(self):
        rf = RF(y, mtry=0.8, n_jobs=2, seed=2001)
        rf.fit(y)

    def test_anytime(self):
        # The feature weights should be as uniformly distributed as possible.
        # We select old features with very little restraint -> we can mess up old distributions,
        # furthermore, we do not correct that -> fulfil at least these loose constraints.

        rf = RF(y, mtry=0.75, n_jobs=30, seed=2001)
        rf.fit(random.randint(0, 10, (5, 20)))
        weights = rf.fit(random.randint(0, 10, (5, 1)))

        self.assertTrue(min(weights) > (mean(weights) - 3*std(weights)))
        self.assertTrue(max(weights) < (mean(weights) + 3*std(weights)))
        assert_almost_equal(mean(weights[0:20]), weights[20], err_msg="The new feature should have the weight equivalent to the average weight of all the previous features")

        weights = rf.fit(random.randint(0, 10, (5, 1)))
        assert_almost_equal(mean(weights[0:21]), weights[21], err_msg="The new feature should have the weight equivalent to the average weight of all the previous features")

    def test_seed(self):
        # When the seeds are the same, the random forest provides identical results.

        rf = RF(y, mtry=0.75, n_jobs=20, seed=2001)
        rf.fit(X1)
        weights1 = rf.fit(X2)

        rf = RF(y, mtry=0.75, n_jobs=20, seed=2001)
        rf.fit(X1)
        weights2 = rf.fit(X2)

        assert_almost_equal(weights1, weights2)

    def test_seed_diff(self):
        ## When the seeds are different, the random forest provides different results.

        rf = RF(y, mtry=0.75, n_jobs=20, seed=2001)
        rf.fit(X1)
        weights1 = rf.fit(X2)

        rf = RF(y, mtry=0.75, n_jobs=20, seed=2002)
        rf.fit(X1)
        weights2 = rf.fit(X2)

        self.assertFalse(all(weights1==weights2))
