import sys
import numpy as np
 
import sys
np.set_printoptions(threshold=sys.maxsize)
file_path = sys.argv[0]
boxes=np.load(file_path)
print(boxes)
