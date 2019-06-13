# Stratified sampling (with or without replacement).
# The point is to make sure that some samples (i.e.: at least 1) from each class get selected.
from numpy import *


def stratified_choice(y, replace):
    # Implemented only for binary labels with {0,1}
    neg = where(y == 0)[0]
    pos = where(y == 1)[0]
    samples_neg = random.choice(neg, size(neg), replace=replace)
    samples_pos = random.choice(pos, size(pos), replace=replace)
    samples = append(samples_neg, samples_pos)
    return samples
