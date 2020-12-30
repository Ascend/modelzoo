from thumt.npu_utils import *
  
##
dtype = tf.float32			# float16 or float32

## model parameters
REFERENCE_NUM = 1       	# the reference pair using in the interface module
TRAIN_DECODE_LENGTH = 61   	# the max sequence length of target sentences
TRAIN_ENCODE_LENGTH = 60	# the max sequence length of source sentences
EVAL_DECODE_LENGTH = 100	# the max sequence length of target sentences
EVAL_ENCODE_LENGTH = 100	# the max sequence length of source sentences
EVAL_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128
TEST_INFERENCE = False
BOS_ID = 2

## NPU configuration
using_NPU = True			# CPU or NPU
using_dynamic = False		# using dynamic shape or static shape
session_config_fn = npu_config if using_NPU else old_session_config


## util settings
STOP_SEE_VARIABLE = False

## Data Path
data_dir = './'#'s3://bi-gru/scripts/thumt/data'
if 's3:' in data_dir:
    import moxing as mox
    mox.file.shift('os', 'mox')

