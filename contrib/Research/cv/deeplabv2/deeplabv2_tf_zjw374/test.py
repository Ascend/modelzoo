import numpy as np

a =[ [[[3,2,1],
       [3,2,1]],
      [[3,2,1],
       [3,2,1]]],
     [[[3,2,1],
       [3,2,1]],
      [[3,2,1],
       [3,2,1]]]]

a = np.array(a)

b = a[:,:,3:]
print(a)
print(b.shape)
print(a.shape)