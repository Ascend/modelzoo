/* *
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ctime>
#include <chrono>
#include <cassert>
#include <pthread.h>
#include <string.h>
#include "common.h"
#include "hw_log.h"
#ifdef VERSION_C75_NOT_C73
#include "acl/acl_prof.h"
#endif
#include "feature_acl.h"
#include "inference_engine.h"
#include "asyn_inference_with_syn_engine.h"
#include "asyn_infernce_with_waitevent_engine.h"
#include "asyn_inference_share_weight_engine.h"
#include "multi_inputs_inference_engine.h"
using namespace std;

std::vector<aclrtContext> contex_vec;

Asyn_Inference_with_syn_engine *Infer = nullptr;
Config *cfg = nullptr;

/* *
 * TestCaseNum: INFERENCE_PROCESSS_ASYN_001
 * Destription: inference async process
 * PreCondition: config json file and om model
 * testProcedure: ./benchmark $jsonFile
 * ExpectedResult: the precision of model is correct
 *   */
TEST_F(ACL, INFERENCE_PROCESSS_ASYN_001)
{
    LOG_INFO("INFERENCE_PROCESSS_ASYN_001 start.");

    aclError ret;

    system("rm -rf ../model1_*");
    system("rm -rf ../model2_*");
    system("rm -rf ./perform_static_*");
    system("rm -rf ./precision*");
    system("rm -rf ./run_*");
    system("rm -rf ./result*");
    system("rm -rf ./ACL_testcase.log");

    uint32_t deviceNum = inference_json_cfg_tbl.commCfg.device_num;

    for (int i = 0; i < deviceNum; i++) {
        if (inference_json_cfg_tbl.commCfg.device_id_vec[i] >= DEVICE_ID_MAX) {
            LOG_ERROR("used max deviceId [%d]  more than limit max[%d]",
                inference_json_cfg_tbl.commCfg.device_id_vec[i], DEVICE_ID_MAX);
            return;
        }
    }

    const char *configPath = "";
    ret = aclInit(configPath);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    ret = SdkInferDeviceContexInit(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);
    LOG_INFO("[step 1] device context initial success");

    uint32_t channelNum = inference_json_cfg_tbl.inferCfg.channelNum;

    Asyn_InferEngine *Infer = new Asyn_InferEngine[deviceNum * channelNum * 2];
    Config *cfg = new Config[deviceNum * channelNum * 2];
    DIR *op = nullptr;

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string rcvPatch1 = inference_json_cfg_tbl.inferCfg.resultFolderPath[0] + "_dev_" +
                std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + "_results";
            op = opendir(rcvPatch1.c_str());
            if (NULL == op) {
                mkdir(rcvPatch1.c_str(), 00775);
            } else {
                closedir(op);
            }

            std::string rcvPatch2 = inference_json_cfg_tbl.inferCfg.resultFolderPath[1] + "_dev_" +
                std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + "_results";
            op = opendir(rcvPatch2.c_str());
            if (NULL == op) {
                mkdir(rcvPatch2.c_str(), 00775);
            } else {
                closedir(op);
            }

            printf("model1 cfg index %d, model2 cfg index %d\n", chnIndex + devIndex * channelNum * 2,
                chnIndex + channelNum + devIndex * channelNum * 2);

            ret =
                GetInferEngineConfig(&(cfg[chnIndex + devIndex * channelNum * 2]), chnIndex + devIndex * channelNum * 2,
                inference_json_cfg_tbl.inferCfg.modelType[0], inference_json_cfg_tbl.dataCfg.dir_path_vec[0], rcvPatch1,
                inference_json_cfg_tbl.inferCfg.omPatch[0], contex_vec[devIndex], inference_json_cfg_tbl);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d GetInferEngineConfig fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = GetInferEngineConfig(&(cfg[chnIndex + channelNum + devIndex * channelNum * 2]),
                chnIndex + channelNum + devIndex * channelNum * 2, inference_json_cfg_tbl.inferCfg.modelType[1],
                inference_json_cfg_tbl.dataCfg.dir_path_vec[1], rcvPatch2, inference_json_cfg_tbl.inferCfg.omPatch[1],
                contex_vec[devIndex], inference_json_cfg_tbl);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d GetInferEngineConfig fail, ret %d", devIndex, (chnIndex + channelNum),
                    ret);
                goto case_end;
            }

            ret = Infer[chnIndex + devIndex * channelNum * 2].Init(&(cfg[chnIndex + devIndex * channelNum * 2]));
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d init fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = Infer[chnIndex + channelNum + devIndex * channelNum * 2].Init(
                &(cfg[chnIndex + channelNum + devIndex * channelNum * 2]));
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d init fail, ret %d", devIndex, (chnIndex + channelNum), ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 2] Infer engine config init and load model success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            ret = Infer[chnIndex + devIndex * channelNum * 2].InferenceThreadProc();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d start inference fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = Infer[chnIndex + channelNum + devIndex * channelNum * 2].InferenceThreadProc();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d start inference fail, ret %d", devIndex, (chnIndex + channelNum), ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 3] Infer engine start success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum * 2].join();
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].join();
        }
    }

    LOG_INFO("[step 4] Infer engine stop success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum * 2].UnloadModel();
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].UnloadModel();

            Infer[chnIndex + devIndex * channelNum * 2].DestroyInferenceMemPool();
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].DestroyInferenceMemPool();
        }
    }

    LOG_INFO("[step 5] Infer engine unload model and release mem pool success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string performFile1 = "./perform_static";
            performFile1 =
                performFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            std::ofstream performStr1(performFile1.c_str(), std::ios::trunc);
            Infer[chnIndex + devIndex * channelNum * 2].DumpTimeCost(performStr1);
            performStr1.close();

            std::string performFile2 = "./perform_static";
            performFile2 = performFile2 + "_dev_" + std::to_string(devIndex) + "_chn_" +
                std::to_string(chnIndex + channelNum) + ".txt";
            std::ofstream performStr2(performFile2.c_str(), std::ios::trunc);
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].DumpTimeCost(performStr2);
            performStr2.close();

            std::string precisionFile1 = "./precision_result";
            precisionFile1 =
                precisionFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum * 2].cfg_->postType == 1) {
                std::ofstream precisionStr1(precisionFile1.c_str(), std::ios::trunc);
                Infer[chnIndex + devIndex * channelNum * 2].CalcTop(precisionStr1);
                precisionStr1.close();
            } else if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum * 2].cfg_->postType == 0) {
                std::string retFolder = Infer[chnIndex + devIndex * channelNum * 2].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType + "/";
                std::string bin2Float = "python3 ../scripts/bintofloat.py ";
                bin2Float = bin2Float + retFolder + " fp32";
                system(bin2Float.c_str());

                std::string runlog = "run_";
                runlog = runlog + to_string(devIndex) + "_" + to_string(chnIndex) + ".log";
                std::string calcTop = "python3 ../scripts/result_statistical.py ";
                calcTop = calcTop + retFolder + " ../datasets/input_1024.csv >> " + runlog;
                system(calcTop.c_str());

                std::string top1Value = "echo \"top1: `cat ";
                top1Value =
                    top1Value + runlog + "|grep \"top1_accuracy_rate\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(top1Value.c_str());

                std::string top5Value = "echo \"top5: `cat ";
                top5Value =
                    top5Value + runlog + "|grep \"top5_accuracy_rate\" | awk '{print $6}'`\" >> " + precisionFile1;
                system(top5Value.c_str());
            }

            std::string precisionFile2 = "./precision_result";
            precisionFile2 = precisionFile2 + "_dev_" + std::to_string(devIndex) + "_chn_" +
                std::to_string(chnIndex + channelNum) + ".txt";

            if (0 == Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->postType == 1) {
                std::ofstream precisionStr2(precisionFile2.c_str(), std::ios::trunc);
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].CalcTop(precisionStr2);
                precisionStr2.close();
            } else if (0 ==
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->postType == 0) {
                std::string retFolder = Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->resultFolder +
                    "/" + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType + "/";
                std::string bin2Float = "python3 ../scripts/bintofloat.py ";
                bin2Float = bin2Float + retFolder + " fp32";
                system(bin2Float.c_str());

                std::string runlog = "run_";
                runlog = runlog + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) + ".log";
                std::string calcTop = "python3 ../scripts/result_statistical.py ";
                calcTop = calcTop + retFolder + " ../datasets/input_1024.csv >> " + runlog;
                system(calcTop.c_str());

                std::string top1Value = "echo \"top1: `cat ";
                top1Value =
                    top1Value + runlog + "|grep \"top1_accuracy_rate\" | awk '{print $4}'`\" >> " + precisionFile2;
                system(top1Value.c_str());

                std::string top5Value = "echo \"top5: `cat ";
                top5Value =
                    top5Value + runlog + "|grep \"top5_accuracy_rate\" | awk '{print $6}'`\" >> " + precisionFile2;
                system(top5Value.c_str());
            }

            if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 4, "yolo")) {
                std::string bin2json = "python3  ../scripts/bintojson.py ";
                bin2json = bin2json + Infer[chnIndex + devIndex * channelNum * 2].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType + "/";
                system(bin2json.c_str());

                std::string calcMap = "python3 ../scripts/detection_map.py  ./result_jsonfile/ "
                    "../datasets/COCO100/HiAIAnnotations  ./ precision.json >>";
                calcMap = calcMap + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) + ".log 2>&1";
                system(calcMap.c_str());

                std::string mapRet = "echo \"map: `cat ";
                mapRet = mapRet + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) +
                    ".log | grep \"Mean AP\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(mapRet.c_str());
            }

            if (0 == Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 4, "yolo")) {
                std::string bin2json = "python3  ../scripts/bintojson.py ";
                bin2json = bin2json + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->resultFolder +
                    "/" + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType + "/";
                system(bin2json.c_str());

                std::string calcMap = "python3 ../scripts/detection_map.py  ./result_jsonfile/ "
                    "../datasets/COCO100/HiAIAnnotations  ./ precision.json >>";
                calcMap = calcMap + "run_" + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) + ".log 2>&1";
                system(calcMap.c_str());


                std::string mapRet = "echo \"map: `cat ";
                mapRet = mapRet + "run_" + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) +
                    ".log | grep \"Mean AP\" | awk '{print $4}'`\" >> " + precisionFile2;
                system(mapRet.c_str());
            }
        }
    }


    LOG_INFO("[step 6] Dump time cost and calculate precision success");

case_end:
    ret = SdkInferDestoryRsc(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    delete[] Infer;
    delete[] cfg;

    LOG_INFO("Success to execute acl inference!");

    return;
}

/* *
 * TestCaseNum: INFERENCE_PROCESSS_SYN_002
 * Destription: inference sync process
 * PreCondition: config json file and om model
 * testProcedure: ./benchmark $jsonFile
 * ExpectedResult: the precision of model is correct
 *   */
TEST_F(ACL, INFERENCE_PROCESSS_SYN_002)
{
    LOG_INFO("INFERENCE_PROCESSS_SYN_002 start.");
    system("rm -rf ../model1_*");
    system("rm -rf ../model2_*");
    system("rm -rf ./perform_static_*");
    system("rm -rf ./precision*");
    system("rm -rf ./run_*");
    system("rm -rf ./result*");
    system("rm -rf ./ACL_testcase.log");
    aclError ret;

    uint32_t deviceNum = inference_json_cfg_tbl.commCfg.device_num;

    for (int i = 0; i < deviceNum; i++) {
        if (inference_json_cfg_tbl.commCfg.device_id_vec[i] >= DEVICE_ID_MAX) {
            LOG_ERROR("used max deviceId [%d]  more than limit max[%d] ",
                inference_json_cfg_tbl.commCfg.device_id_vec[i], DEVICE_ID_MAX);
            return;
        }
    }

    const char *configPath = "";
    ret = aclInit(configPath);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    ret = SdkInferDeviceContexInit(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);
    LOG_INFO("[step 1] device context initial success");

    uint32_t channelNum = inference_json_cfg_tbl.inferCfg.channelNum;

    InferEngine *Infer = new InferEngine[deviceNum * channelNum * 2];
    Config *cfg = new Config[deviceNum * channelNum * 2];
    DIR *op = nullptr;

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string rcvPatch1 = inference_json_cfg_tbl.inferCfg.resultFolderPath[0] + "_dev_" +
                std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + "_results";
            op = opendir(rcvPatch1.c_str());
            if (NULL == op) {
                mkdir(rcvPatch1.c_str(), 00775);
            } else {
                closedir(op);
            }

            std::string rcvPatch2 = inference_json_cfg_tbl.inferCfg.resultFolderPath[1] + "_dev_" +
                std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + "_results";
            op = opendir(rcvPatch2.c_str());
            if (NULL == op) {
                mkdir(rcvPatch2.c_str(), 00775);
            } else {
                closedir(op);
            }

            printf("model1 cfg index %d, model2 cfg index %d\n", chnIndex + devIndex * channelNum * 2,
                chnIndex + channelNum + devIndex * channelNum * 2);

            ret =
                GetInferEngineConfig(&(cfg[chnIndex + devIndex * channelNum * 2]), chnIndex + devIndex * channelNum * 2,
                inference_json_cfg_tbl.inferCfg.modelType[0], inference_json_cfg_tbl.dataCfg.dir_path_vec[0], rcvPatch1,
                inference_json_cfg_tbl.inferCfg.omPatch[0], contex_vec[devIndex], inference_json_cfg_tbl);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d GetInferEngineConfig fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = GetInferEngineConfig(&(cfg[chnIndex + channelNum + devIndex * channelNum * 2]),
                chnIndex + channelNum + devIndex * channelNum * 2, inference_json_cfg_tbl.inferCfg.modelType[1],
                inference_json_cfg_tbl.dataCfg.dir_path_vec[1], rcvPatch2, inference_json_cfg_tbl.inferCfg.omPatch[1],
                contex_vec[devIndex], inference_json_cfg_tbl);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d GetInferEngineConfig fail, ret %d", devIndex, (chnIndex + channelNum),
                    ret);
                goto case_end;
            }

            ret = Infer[chnIndex + devIndex * channelNum * 2].Init(&(cfg[chnIndex + devIndex * channelNum * 2]));
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d init fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = Infer[chnIndex + channelNum + devIndex * channelNum * 2].Init(
                &(cfg[chnIndex + channelNum + devIndex * channelNum * 2]));
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d init fail, ret %d", devIndex, (chnIndex + channelNum), ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 2] Infer engine config init and load model success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            ret = Infer[chnIndex + devIndex * channelNum * 2].InferenceThreadProc();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d start inference fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = Infer[chnIndex + channelNum + devIndex * channelNum * 2].InferenceThreadProc();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model2 dev %d chn %d start inference fail, ret %d", devIndex, (chnIndex + channelNum), ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 3] Infer engine start success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum * 2].join();
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].join();
        }
    }

    LOG_INFO("[step 4] Infer engine stop success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum * 2].UnloadModel();
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].UnloadModel();
        }
    }

    LOG_INFO("[step 5] Infer engine unload model and release mem pool success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string performFile1 = "./perform_static";
            performFile1 =
                performFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            std::ofstream performStr1(performFile1.c_str(), std::ios::trunc);
            Infer[chnIndex + devIndex * channelNum * 2].DumpTimeCost(performStr1);
            performStr1.close();

            std::string performFile2 = "./perform_static";
            performFile2 = performFile2 + "_dev_" + std::to_string(devIndex) + "_chn_" +
                std::to_string(chnIndex + channelNum) + ".txt";
            std::ofstream performStr2(performFile2.c_str(), std::ios::trunc);
            Infer[chnIndex + channelNum + devIndex * channelNum * 2].DumpTimeCost(performStr2);
            performStr2.close();

            std::string precisionFile1 = "./precision_result";
            precisionFile1 =
                precisionFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum * 2].cfg_->postType == 1) {
                std::ofstream precisionStr1(precisionFile1.c_str(), std::ios::trunc);
                Infer[chnIndex + devIndex * channelNum * 2].CalcTop(precisionStr1);
                precisionStr1.close();
            } else if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum * 2].cfg_->postType == 0) {
                std::string retFolder = Infer[chnIndex + devIndex * channelNum * 2].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType + "/";
                std::string bin2Float = "python3 ../scripts/bintofloat.py ";
                bin2Float = bin2Float + retFolder + " fp32";
                system(bin2Float.c_str());

                std::string runlog = "run_";
                runlog = runlog + to_string(devIndex) + "_" + to_string(chnIndex) + ".log";
                std::string calcTop = "python3 ../scripts/result_statistical.py ";
                calcTop = calcTop + retFolder + " ../datasets/input_1024.csv >> " + runlog;
                system(calcTop.c_str());

                std::string top1Value = "echo \"top1: `cat ";
                top1Value =
                    top1Value + runlog + "|grep \"top1_accuracy_rate\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(top1Value.c_str());

                std::string top5Value = "echo \"top5: `cat ";
                top5Value =
                    top5Value + runlog + "|grep \"top5_accuracy_rate\" | awk '{print $6}'`\" >> " + precisionFile1;
                system(top5Value.c_str());
            }

            std::string precisionFile2 = "./precision_result";
            precisionFile2 = precisionFile2 + "_dev_" + std::to_string(devIndex) + "_chn_" +
                std::to_string(chnIndex + channelNum) + ".txt";

            if (0 == Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->postType == 1) {
                std::ofstream precisionStr2(precisionFile2.c_str(), std::ios::trunc);
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].CalcTop(precisionStr2);
                precisionStr2.close();
            } else if (0 ==
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->postType == 0) {
                std::string retFolder = Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->resultFolder +
                    "/" + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType + "/";
                std::string bin2Float = "python3 ../scripts/bintofloat.py ";
                bin2Float = bin2Float + retFolder + " fp32";
                system(bin2Float.c_str());

                std::string runlog = "run_";
                runlog = runlog + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) + ".log";
                std::string calcTop = "python3 ../scripts/result_statistical.py ";
                calcTop = calcTop + retFolder + " ../datasets/input_1024.csv >> " + runlog;
                system(calcTop.c_str());

                std::string top1Value = "echo \"top1: `cat ";
                top1Value =
                    top1Value + runlog + "|grep \"top1_accuracy_rate\" | awk '{print $4}'`\" >> " + precisionFile2;
                system(top1Value.c_str());

                std::string top5Value = "echo \"top5: `cat ";
                top5Value =
                    top5Value + runlog + "|grep \"top5_accuracy_rate\" | awk '{print $6}'`\" >> " + precisionFile2;
                system(top5Value.c_str());
            }

            if (0 == Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType.compare(0, 4, "yolo")) {
                std::string bin2json = "python3  ../scripts/bintojson.py ";
                bin2json = bin2json + Infer[chnIndex + devIndex * channelNum * 2].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType + "/";
                system(bin2json.c_str());

                std::string calcMap = "python3 ../scripts/detection_map.py  ./result_jsonfile/ "
                    "../datasets/COCO100/HiAIAnnotations  ./ precision.json >>";
                calcMap = calcMap + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) + ".log 2>&1";
                system(calcMap.c_str());

                std::string mapRet = "echo \"map: `cat ";
                mapRet = mapRet + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) +
                    ".log | grep \"Mean AP\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(mapRet.c_str());
            }

            if (0 == Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType.compare(0, 4, "yolo")) {
                std::string bin2json = "python3  ../scripts/bintojson.py ";
                bin2json = bin2json + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->resultFolder +
                    "/" + Infer[chnIndex + channelNum + devIndex * channelNum * 2].cfg_->modelType + "/";
                system(bin2json.c_str());

                std::string calcMap = "python3 ../scripts/detection_map.py  ./result_jsonfile/ "
                    "../datasets/COCO100/HiAIAnnotations  ./ precision.json >>";
                calcMap = calcMap + "run_" + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) + ".log 2>&1";
                system(calcMap.c_str());


                std::string mapRet = "echo \"map: `cat ";
                mapRet = mapRet + "run_" + to_string(devIndex) + "_" + to_string(chnIndex + channelNum) +
                    ".log | grep \"Mean AP\" | awk '{print $4}'`\" >> " + precisionFile2;
                system(mapRet.c_str());
            }
        }
    }
    LOG_INFO("[step 6] Dump time cost and calculate precision success");

case_end:
    ret = SdkInferDestoryRsc(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    delete[] Infer;
    delete[] cfg;

    LOG_INFO("Success to execute acl inference!");

    return;
}

/* *
 * TestCaseNum: INFERENCE_PROCESSS_SYN_MULTI_INPUTS_005
 * Destription: multiple device and multiple channel inference
 * PreCondition: config json file and om model
 * testProcedure: ./benchmark $jsonFile
 * ExpectedResult: the precision of model is correct
 *   */
TEST_F(ACL, INFERENCE_PROCESSS_SYN_MULTI_INPUTS_005)
{
    LOG_INFO("INFERENCE_PROCESSS_SYN_004 start.");
    printf("INFERENCE_PROCESSS_SYN_004 start.");
    system("rm -rf ../model1_*");
    system("rm -rf ../model2_*");
    system("rm -rf ./perform_static_*");
    system("rm -rf ./precision*");
    system("rm -rf ./run_*");
    system("rm -rf ./result*");
    system("rm -rf ./ACL_testcase.log");

    aclError ret;
    uint32_t deviceNum = inference_json_cfg_tbl.commCfg.device_num;

    for (int i = 0; i < deviceNum; i++) {
        if (inference_json_cfg_tbl.commCfg.device_id_vec[i] >= DEVICE_ID_MAX) {
            LOG_ERROR("used max deviceId [%d]  more than limit max[%d]",
                inference_json_cfg_tbl.commCfg.device_id_vec[i], DEVICE_ID_MAX);
            return;
        }
    }

    const char *configPath = "";
    ret = aclInit(configPath);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    ret = SdkInferDeviceContexInit(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);
    LOG_INFO("[step 2===] device context initial success");

    uint32_t channelNum = inference_json_cfg_tbl.inferCfg.channelNum;

    multi_inputs_Inference_engine *Infer = new multi_inputs_Inference_engine[deviceNum * channelNum];
    Config *cfg = new Config[deviceNum * channelNum];

    DIR *op = nullptr;
    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string rcvPatch1 = inference_json_cfg_tbl.inferCfg.resultFolderPath[0] + "_dev_" +
                std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + "_results";
            op = opendir(rcvPatch1.c_str());
            if (NULL == op) {
                mkdir(rcvPatch1.c_str(), 00775);
            } else {
                closedir(op);
            }

            LOG_INFO("model1 cfg index %d \n", chnIndex + devIndex * channelNum);

            ret = GetInferEngineConfig(&(cfg[chnIndex + devIndex * channelNum]), chnIndex + devIndex * channelNum,
                inference_json_cfg_tbl.inferCfg.modelType[0], inference_json_cfg_tbl.dataCfg.dir_path_vec[0], rcvPatch1,
                inference_json_cfg_tbl.inferCfg.omPatch[0], contex_vec[devIndex], inference_json_cfg_tbl);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d GetInferEngineConfig fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }

            ret = Infer[chnIndex + devIndex * channelNum].Init(&(cfg[chnIndex + devIndex * channelNum]));
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d init fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 2] Infer engine config init and load model success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            ret = Infer[chnIndex + devIndex * channelNum].Multi_Inputs_InferenceThreadProc();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("model1 dev %d chn %d start inference fail, ret %d", devIndex, chnIndex, ret);
                goto case_end;
            }
        }
    }

    LOG_INFO("[step 3] Infer engine start success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum].join();
        }
    }

    LOG_INFO("[step 4] Infer engine stop success");

    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            Infer[chnIndex + devIndex * channelNum].UnloadModel();
        }
    }

    LOG_INFO("[step 5] Infer engine unload model and release mem pool success");
    for (uint32_t devIndex = 0; devIndex < deviceNum; devIndex++) {
        for (uint32_t chnIndex = 0; chnIndex < channelNum; chnIndex++) {
            std::string performFile1 = "./perform_static";
            performFile1 =
                performFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            std::ofstream performStr1(performFile1.c_str(), std::ios::trunc);
            Infer[chnIndex + devIndex * channelNum].DumpTimeCost(performStr1);
            performStr1.close();

            std::string precisionFile1 = "./precision_result";
            precisionFile1 =
                precisionFile1 + "_dev_" + std::to_string(devIndex) + "_chn_" + std::to_string(chnIndex) + ".txt";
            if (0 == Infer[chnIndex + devIndex * channelNum].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum].cfg_->postType == 1) {
                std::ofstream precisionStr1(precisionFile1.c_str(), std::ios::trunc);
                Infer[chnIndex + devIndex * channelNum].CalcTop(precisionStr1);
                precisionStr1.close();
            } else if (0 == Infer[chnIndex + devIndex * channelNum].cfg_->modelType.compare(0, 6, "resnet") &&
                Infer[chnIndex + devIndex * channelNum].cfg_->postType == 0) {
                std::string retFolder = Infer[chnIndex + devIndex * channelNum].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum].cfg_->modelType + "/";
                std::string bin2Float = "python3 ../scripts/bintofloat.py ";
                bin2Float = bin2Float + retFolder + " fp32";
                system(bin2Float.c_str());

                std::string runlog = "run_";
                runlog = runlog + to_string(devIndex) + "_" + to_string(chnIndex) + ".log";
                std::string calcTop = "python3 ../scripts/result_statistical.py ";
                calcTop = calcTop + retFolder + " ../datasets/input_1024.csv >> " + runlog;
                system(calcTop.c_str());

                std::string top1Value = "echo \"top1: `cat ";
                top1Value =
                    top1Value + runlog + "|grep \"top1_accuracy_rate\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(top1Value.c_str());

                std::string top5Value = "echo \"top5: `cat ";
                top5Value =
                    top5Value + runlog + "|grep \"top5_accuracy_rate\" | awk '{print $6}'`\" >> " + precisionFile1;
                system(top5Value.c_str());
            }

            if (0 == Infer[chnIndex + devIndex * channelNum].cfg_->modelType.compare(0, 4, "yolo")) {
                std::string bin2json = "python3  ../scripts/bintojson.py ";
                bin2json = bin2json + Infer[chnIndex + devIndex * channelNum * 2].cfg_->resultFolder + "/" +
                    Infer[chnIndex + devIndex * channelNum * 2].cfg_->modelType + "/";
                system(bin2json.c_str());

                std::string calcMap = "python3 ../scripts/detection_map.py  ./result_jsonfile/ "
                    "../datasets/COCO100/HiAIAnnotations  ./ precision.json >>";
                calcMap = calcMap + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) + ".log 2>&1";
                system(calcMap.c_str());

                std::string mapRet = "echo \"map: `cat ";
                mapRet = mapRet + "run_" + to_string(devIndex) + "_" + to_string(chnIndex) +
                    ".log | grep \"Mean AP\" | awk '{print $4}'`\" >> " + precisionFile1;
                system(mapRet.c_str());
            }
        }
    }
    LOG_INFO("[step 6] Dump time cost and calculate precision success");

case_end:
    ret = SdkInferDestoryRsc(inference_json_cfg_tbl.commCfg.device_id_vec, contex_vec);
    EXPECT_EQ(ACL_ERROR_NONE, ret);

    delete[] Infer;
    delete[] cfg;

    LOG_INFO("Success to execute acl inference!");

    return;
}
