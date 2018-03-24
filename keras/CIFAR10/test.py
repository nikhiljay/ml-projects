import numpy as np
import operator

test = [1, 2, 8, 4, 5, 6]
index, value = max(enumerate(test), key=operator.itemgetter(1))
print(str(index) + ' and ' + str(value*100) + "% confidence")
