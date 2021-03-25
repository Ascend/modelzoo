import os
import numpy as np
import pandas as pd
import tqdm
tmp0 = []
files = os.listdir('./output5/output5/20210321_153930')
for file in files:
	tmp1 = []
	f = open("./output5/output5/20210321_153930/"+file) #  0.8993736039103426 Test318 200.89.035
	for line in f:
		for i in line.split():
			tmp1.append(float(i))
	f.close()
	tmp1 = np.array(tmp1)
	tmp1 = tmp1.reshape((1, 4, 64, 64, 16))
	tmp0.append(tmp1)
	print(len(tmp0))
label_0 = []
files_2 = os.listdir('./output/output/label')
for file in files_2:	
	if file.endswith('bin'):
		tmp2 = np.fromfile('./output/output/label/'+file,dtype = 'float32')
		print(tmp2.shape)
		tmp2 = tmp2.reshape((1, 4, 64, 64, 16))
		print(tmp2.shape)
		label_0.append(tmp2)

def _accuracy_computation(output,gtMaps):
		""" Computes accuracy tensor
		"""
		joint_accur = []
		for i in range(16):
			joint_accur.append(_accur(output[:, 3, :, :,i], gtMaps[:, 3, :, :, i], 1))
		# print(sum(joint_accur)/16)
		return sum(joint_accur)/16

def _accur(pred, gtMap, num_image):
		""" Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
		returns one minus the mean distance.
		Args:
			pred		: Prediction Batch (shape = num_image x 64 x 64)
			gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
			num_image 	: (int) Number of images in batch
		Returns:
			(float)
		"""
		err  = 0.0
		for i in range(num_image):
			err += _compute_err(pred[i], gtMap[i])
		return (1-err)/num_image
def _compute_err(u, v):
		""" Given 2 tensors compute the euclidean distance (L2) between maxima locations
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
		u_x,u_y = _argmax(u)
		v_x,v_y = _argmax(v)
		return (((u_x - v_x) *(u_x - v_x)+ (u_y - v_y)*(u_y - v_y))** 0.5)/91
def _argmax(tensor):

		""" ArgMax
		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position
		"""
		# resh = tf.reshape(tensor, [-1])
		# print('shape: ',tensor[0])
		resh = np.array(tensor).reshape([-1]).tolist()
		argmax = resh.index(max(resh))
		return (argmax // np.array(tensor).shape[0], argmax % np.array(tensor).shape[0])
# assert len(tmp0) == len(label_0)
accuracy = 0
for i in range(len(tmp0)):
	acc = _accuracy_computation(tmp0[i],label_0[i])
	accuracy += acc
	print(i)
	print(accuracy/(i+1))
print("final",accuracy/len(tmp0))