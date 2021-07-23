import json
from StreamManagerApi import *
import cv2
if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("/root/yzr/mindspore_modelzoo/CenterFace/infer/data/config/centerface_no_aipp.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    dataInput.data=open("test.jpg","rb").read()

    # Inputs data to a specified stream based on streamName.
    streamName = b"im_centerface"
    inPluginId = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    # print the infer result

    res=(inferResult.data.decode())
    print(json.loads(res))
    # destroy streams
    streamManagerApi.DestroyAllStreams()
