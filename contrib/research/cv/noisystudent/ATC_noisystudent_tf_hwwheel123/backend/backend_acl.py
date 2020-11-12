"""
acl backend
"""


import dnmetis_backend as dnmetis_backend
import backend.backend as backend
import numpy as np
import os
import pdb

class AclBackend(backend.Backend):
    def __init__(self):
        super(AclBackend, self).__init__()
        self.ACL=5
        self.outputs = ""
        self.inputs = ""
        self.model_path = ""
        self.cfg_path = ""

    def version(self):
        return "1.0"

    def name(self):
        return "AclBackend"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self, args):
        # there is no input/output meta data i the graph so it need to come from config.
        if not args.inputs:
            raise ValueError("AclBackend needs inputs")
        if not args.outputs:
            raise ValueError("AclBackend needs outputs")
        self.outputs = args.outputs
        self.inputs = args.inputs
        self.model_path = args.model
        self.cfg_path = args.cfg_path
        #s.path.join(args.pwd, 'backend_cfg/built-in_config.txt')
        dnmetis_backend.backend_setconfig(self.cfg_path)
        dnmetis_backend.backend_load(self.ACL,self.model_path,"")
        return self

    def predict(self, feed):
        #fed=feed[self.inputs[0]]
        result_list=[]
        result = dnmetis_backend.backend_predict(self.ACL,self.model_path,feed)

        for _ in range(len(self.outputs)):
            #resnet50 tf & caffe
            if 'softmax_tensor' in self.outputs[_] or 'prob' in self.outputs[_]:
                result_list.append(np.argmax(result[_],1))
            # resnet50 tf
            if 'ArgMax' in self.outputs[_]:
                result_list.append(result[_])
        if result_list == []:
            # ssd-resnet34 tf
            result_list = result
        return result_list

    def unload(self):
        return dnmetis_backend.backend_unload(self.ACL,self.model_path,"")
