"""Frozen one-axis schedule selector; H30 delegates the original arithmetic."""
import math
from a1_allocation_loss import learning_rate as original_learning_rate


def scheduled_learning_rate(update, lr, per_epoch):
    horizon = lr.get('horizon', 30)
    if horizon == 30:
        return original_learning_rate(update, lr, per_epoch)
    if horizon != 20 or type(update) is not int or update < 1 or per_epoch != 2952:
        raise ValueError('unregistered LR horizon/update geometry')
    epoch = min(max(math.ceil((update - 5) / per_epoch), 1), 20)
    return lr['end'] + .5 * (lr['start'] - lr['end']) * (1 + math.cos(math.pi * (epoch - 1) / 19))
