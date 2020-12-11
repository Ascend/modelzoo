# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CI/CD for vega."""
import vega
import os
from vega.core.common.utils import update_dict
from vega.core.common import Config
import pandas as pd
import time
import subprocess
import traceback
import datetime


def main():
    """Execute main."""
    algs = Config("./ci.yml")
    run_status = pd.DataFrame({"algorithms": [], "run_time(min)": [], "status": [], "error_info": []})
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    csv_name = 'run_status_' + time_stamp + ".csv"
    task_status = "Sucess"
    for alg in algs.keys():
        print("{} start.".format(alg))
        error_info = None
        try:
            current_path = os.path.abspath(os.path.join(os.getcwd(), "../", "vega-di-g"))
            alg_cfg = Config(current_path + algs[alg].cfg_file)
            if 'args' in algs[alg].keys():
                update_dict(algs[alg].args, alg_cfg)

        except:
            print("{} end.".format(alg))
            elapsed_time = 0
            error_info = traceback.format_exc()
            status = "Fail"
            status_info = error_info
            run_info = {"algorithms": alg, "run_time(min)": elapsed_time, "status": status, "error_info": status_info}
            run_status = run_status.append(run_info, ignore_index=True)
            run_status.to_csv(csv_name)

        else:
            try:
                start_time = time.clock()

                if alg == "simple_cnn_tf":
                    from nas_tf.simple_cnn.simple_rand import SimpleRand

                vega.run(alg_cfg)

            except:
                error_info = traceback.format_exc()

            finally:
                end_time = time.clock()
                print("{} end.".format(alg))
                elapsed_time = int((end_time - start_time) / 60)
                if error_info is None:
                    try:
                        error_info = subprocess.check_output(["bash", "./utils/split_log.sh", alg + " start.",
                                                              alg + " end.", alg + ".log"])
                    except:
                        error_info = None

                status = "Sucess" if error_info is None else "Fail"
                if status == "Fail":
                    task_status = "Fail"
                status_info = "" if error_info is None else error_info
                run_info = {"algorithms": alg, "run_time(min)": elapsed_time, "status": status,
                            "error_info": status_info}
                run_status = run_status.append(run_info, ignore_index=True)
                run_status.to_csv(csv_name)

    if task_status == "Sucess":
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
