import numpy as np
import math

class TinyStatistician:
    def __init__(self):
        return
    

    def mean(self, x):
        if (isinstance(x, list) and x) or (isinstance(x, np.ndarray) and x.size > 0):
            mean = float(sum(val for val in x) / len(x))
        else:
            return None
        return mean

    def median(self, x):
        if (isinstance(x, list) and x) or (isinstance(x, np.ndarray) and x.size > 0):
            y = x
            y.sort()
            if len(x) % 2 == 1:
                return float(y[len(y) // 2])
            else:
                return (self.mean([y[(len(y) // 2) - 1], y[len(y) // 2]]))
        else:
            return None

    def quartiles(self, x):
        if (isinstance(x, list) and x) or (isinstance(x, np.ndarray) and x.size > 0):
            y = x
            y.sort()
            if len(y) % 2 == 1:
                left = x[:(len(y) // 2)]
                right = x[(len(y) // 2) + 1:]
            else:
                left = x[:(len(y) // 2)]
                right = x[(len(y) // 2):]
            return [self.median(left), self.median(right)]
        
    def percentile(self, x, p):
        y = x
        y.sort()
        i = (p / 100) * (len(y) - 1)
        if (int(i) == i):
            return y[i]
        else:
            if (int(i) > i):
                return (y[int(i) - 1] + (y[int(i)] - y[int(i) - 1]) * i)
            else:
                return (y[int(i)] + (y[int(i) + 1] - y[int(i)]) * i)

    def var(self, x):
        m = self.mean(x)
        var = float(sum(pow(val - m, 2) for val in x) / (len(x)))
        return var
    
    def std(self, x):
        return (math.sqrt(self.var(x)))

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    stat = TinyStatistician()
    if x.shape == (1, 1):
        return (x - stat.mean(x)) / stat.std(x)
    else:
        return (x.flatten() - stat.mean(x)) / stat.std(x)