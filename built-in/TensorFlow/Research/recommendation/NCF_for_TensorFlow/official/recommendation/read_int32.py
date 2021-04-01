import numpy as np
import sys

inputfile= sys.argv[1]
outputfile = sys.argv[2]

data = np.fromfile(inputfile, dtype = np.int32)
np.savetxt(outputfile, np.reshape(data, (-1, 1)), delimiter = ' ', fmt="%.8f")
