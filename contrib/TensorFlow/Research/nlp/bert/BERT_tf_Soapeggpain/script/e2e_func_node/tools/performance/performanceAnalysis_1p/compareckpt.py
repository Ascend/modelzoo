import os
import numpy as np
from tensorflow.python import pywrap_tensorflow


class CompareCkpt:
    def __init__(self, input_path=None):
        """
        :param input_path: result file absolute path, like '/e2e_test_new/result/exec-20190820100329'
        """
        self.input_path = input_path
        self.compare_ckpt()

    @staticmethod
    def get_ckpt_list(input_path):
        ckpt_path_list = []
        for root, dirs, files in os.walk(input_path):
            for ckpt_dir in dirs:
                if 'ckpt' in ckpt_dir:
                    ckpt_path_list.append(os.path.join(root, ckpt_dir))
        return ckpt_path_list

    @staticmethod
    def get_file_list(ckpt_path):
        ckpt_file_list = []
        for root, dirs, files in os.walk(ckpt_path):
            for ckpt_file in files:
                if 'index' in ckpt_file:
                    ckpt_file_list.append(os.path.join(root, ckpt_file).split('.index')[0])
        return ckpt_file_list

    def compare_ckpt(self):
        ckpt_path_list = CompareCkpt.get_ckpt_list(self.input_path)
        ckpt_file_list = []
        for ckpt_path in ckpt_path_list:
            ckpt_file_list.append(CompareCkpt.get_file_list(ckpt_path))
        for i in range(len(ckpt_file_list) - 1):
            if len(ckpt_file_list[i]) != len(ckpt_file_list[i + 1]):
                raise ValueError('check point file total number is not equal.')
            for j in range(len(ckpt_file_list[i])):
                reader_1st = pywrap_tensorflow.NewCheckpointReader(ckpt_file_list[i][j])
                var_1st = reader_1st.get_variable_to_shape_map()
                reader_2nd = pywrap_tensorflow.NewCheckpointReader(ckpt_file_list[i + 1][j])
                var_2nd = reader_2nd.get_variable_to_shape_map()
                error_flag = False
                for key in var_1st:
                    if 'moving_mean' not in key and 'moving_variance' not in key:
                        if var_1st[key] != var_2nd[key]:
                            print(key + ' in check point file ' +
                                  str(ckpt_file_list[i][j]).split("\\")[-1] + ' is not the same with ' +
                                  str(ckpt_file_list[i + 1][j]).split("\\")[-1])
                            error_flag = True
                        else:
                            value_1st = reader_1st.get_tensor(key)
                            value_2nd = reader_2nd.get_tensor(key)
                            if np.allclose(value_1st, value_2nd, 0, 0):
                                pass
                            else:
                                print(key + ' in check point file ' +
                                      str(ckpt_file_list[i][j]).split("\\")[-1] + ' is not the same with ' +
                                      str(ckpt_file_list[i + 1][j]).split("\\")[-1])
                                error_flag = True
                if error_flag:
                    pass
                else:
                    print('check point file ' + str(ckpt_file_list[i][j]).split("\\")[-1] + ' is the same with ' +
                          str(ckpt_file_list[i + 1][j]).split("\\")[-1])
