/*
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
Created by wang-bain on 2021/3/18.
*/


#include "utils.h"

using namespace std;

uint32_t modelId = 0;
uint32_t stepNumber = 0;
float totalSamplesTime = 0;

int main(int argc, char **argv) {
    Result ret;
    map<char, string> params;

    ret = InitAndCheckParams(argc, argv, params);
    if (ret != SUCCESS) {
        return FAILED;
    }

    vector<aclrtContext> v_context;
    // ACL接口初始化，包含aclInit，aclrtSetDevice，aclrtCreateContext等接口
    ret = InitResource(v_context, params);
    if (ret != SUCCESS) {
        printf("%s - E - [XACL]: Init acl resource failed\n",
               GetTime().c_str());
        return FAILED;
    } else {
        printf("%s - I - [XACL]: Init acl resource success\n",
               GetTime().c_str());
    }

    // 加载om模型文件，包含aclmdlLoadFromFile等接口
    ret = LoadModelFromFile(&modelId, params);
    if (ret != SUCCESS) {
        printf("%s - E - [XACL]: Load acl model from file failed\n",
               GetTime().c_str());
        return FAILED;
    } else {
        printf("%s - I - [XACL]: Load acl model from file success\n",
               GetTime().c_str());
    }

    // 创建模型描述，包含aclmdlCreateDesc，aclmdlGetDesc等接口
    aclmdlDesc *modelDesc = nullptr;
    ret = CreateModelDesc(modelDesc, modelId);
    if (ret != SUCCESS) {
        printf("%s - E - [XACL]: Create model description failed\n",
               GetTime().c_str());
        // 创建模型描述失败时，释放模型，模型描述和模型上下文，包含ModelUnloadAndDescDestroy，DeviceContextDestroy等接口
        DestroyResource(modelId, modelDesc, v_context, params);
        return FAILED;
    } else {
        printf("%s - I - [XACL]: Create model description success\n",
               GetTime().c_str());
    }
    // 输入文件列表
    vector<string> v_inputFiles;
    vector<vector<string>> saveOriginInput;
    Split(params['i'], v_inputFiles, ",");
    if ((!v_inputFiles.empty()) && (v_inputFiles[0].find(".bin") == string::npos)) {
        // 输入是目录时，遍历目录下所有输入文件
        printf("%s - I - [XACL]: Input type is director\n",
               GetTime().c_str());
        vector <vector<string>> v_allInputFiles;
        for (uint32_t fileIdx = 0; fileIdx < v_inputFiles.size(); ++fileIdx) {
            vector <string> v_fileName;
            ScanFiles(v_fileName, v_inputFiles[fileIdx]);
            sort(v_fileName.begin(), v_fileName.end());
            v_allInputFiles.push_back(v_fileName);
        }
        // 当输入是目录时，判断是否需要拼接输入
        if (params['g'] != "0") {
            // 当输入是目录时，且按Batch拼接输入标志为true，先拼接文件存放到拼接后目录
            // 创建拼接后文件目录，命名规则为：原文件目录 + 'batch' + N
            vector <string> v_mergedInputPath;
            for (uint32_t inputIdx = 0; inputIdx < v_inputFiles.size(); ++inputIdx) {
                string inputIdxPath;
                if (v_inputFiles[inputIdx].find("/",v_inputFiles[inputIdx].length() - 1) == string::npos) {
                    inputIdxPath = v_inputFiles[inputIdx] + "_" + params['b'];
                } else {
                    inputIdxPath = v_inputFiles[inputIdx].substr(0, v_inputFiles[inputIdx].length() -1) + "_" + params['b'];
                }

                v_mergedInputPath.push_back(inputIdxPath);
                ret = CreateDir(inputIdxPath);
                if (ret != SUCCESS) {
                    return FAILED;
                }
            }

            vector <vector<string>> v_allMergedInputFiles;
            // 循环每一个输入
            for (uint32_t inputIdx = 0; inputIdx < v_allInputFiles.size(); ++inputIdx) {
                vector <string> v_inputFileName;
                Split(v_inputFiles[inputIdx], v_inputFileName, "/");
                string inputName = v_inputFileName[v_inputFileName.size() - 1];

                printf("%s - I - [XACL]: Start to merge input %s\n",
                       GetTime().c_str(), inputName.c_str());
                uint32_t fileIdx = 0, sliceNum = 0;
                vector <string> v_mergedInputFiles;
                for (; fileIdx < v_allInputFiles[0].size(); fileIdx += atoi(params['b'].c_str()), ++sliceNum) {
                    // 若剩余文件不足一个batch则丢弃
                    if ((sliceNum + 1) * atoi(params['b'].c_str()) > v_allInputFiles[0].size()) {
                        break;
                    }
                    vector <string> v_mergeFiles;
                    string mergedNum = FormatInt(to_string(sliceNum), 5);
                    string mergedFile = v_mergedInputPath[inputIdx] + "/" + inputName + "_" + mergedNum + ".bin";
                    v_mergedInputFiles.push_back(mergedFile);
                    for (uint32_t b = 0; b < atoi(params['b'].c_str()); ++b) {
                        v_mergeFiles.push_back(v_allInputFiles[inputIdx][fileIdx + b]);
                    }
                    MergeInputFile(v_mergeFiles, mergedFile);
                    saveOriginInput.push_back(v_mergeFiles);
                }
                v_allMergedInputFiles.push_back(v_mergedInputFiles);
                printf("%s - I - [XACL]: Merge input %s to %s finished\n",
                       GetTime().c_str(), inputName.c_str(), v_mergedInputPath[inputIdx].c_str());

            }
            // 将拼接后的输入目录替换原始输入目录
            v_allInputFiles = v_allMergedInputFiles;
        }

        // 循环执行输入目录下所有文件
        for (uint32_t fileIdx = 0; fileIdx < v_allInputFiles[0].size(); ++fileIdx) {
            vector <string> v_singleInputFiles;
            for (uint32_t inputIdx = 0; inputIdx < v_allInputFiles.size(); ++inputIdx) {
                printf("%s - I - [XACL]: The input file: %s is checked\n",
                       GetTime().c_str(), v_allInputFiles[inputIdx][fileIdx].c_str());
                v_singleInputFiles.push_back(v_allInputFiles[inputIdx][fileIdx]);
            }

            // 创建输入
            aclmdlDataset *input = nullptr;
            string outputPrefix;
            ret = CreateInput(modelDesc, input, outputPrefix, v_singleInputFiles, saveOriginInput, params);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Create input data failed\n",
                       GetTime().c_str());
                DestroyResource(modelId, modelDesc, v_context, params);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create input data success\n",
                       GetTime().c_str());
            }

            // 动态shape适配
            if (params['v'] != "") {
                if (params['w'] != "") {
                    aclTensorDesc *inputDesc[v_singleInputFiles.size()];
                    ret = CreateInputDesc(input, v_singleInputFiles, inputDesc, params);
                    if (ret != SUCCESS) {
                        printf("%s - E - [XACL]: Create input description failed\n",
                               GetTime().c_str());
                        DestroyResource(modelId, modelDesc, v_context, params);
                        return FAILED;
                    } else {
                        printf("%s - I - [XACL]: Create input description success\n",
                               GetTime().c_str());
                    }

                } else {
                    printf("%s - E - [XACL]: When dynamicShape function is enable, must set dynamicOutSize\n",
                           GetTime().c_str());
                    DestroyResource(modelId, modelDesc, v_context, params);
                    return FAILED;
                }
            }

            // 创建输出
            aclmdlDataset *output = nullptr;
            ret = CreateOutput(modelDesc, output, input, params);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Create output data failed\n",
                       GetTime().c_str());
                DestroyResource(modelId, modelDesc, v_context, params);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create output data success\n",
                       GetTime().c_str());
            }

            // 执行推理
            ret = Execute(input, output, modelDesc, modelId, v_context, totalSamplesTime, outputPrefix, saveOriginInput, fileIdx, params);

            stepNumber++;

            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Execute acl model failed\n",
                       GetTime().c_str());
                DestroyResource(modelId, modelDesc, v_context, params);
                return FAILED;
            }
        }
        DestroyResource(modelId, modelDesc, v_context, params);
    } else {
        // 创建输入
        aclmdlDataset *input = nullptr;
        string outputPrefix;
        ret = CreateInput(modelDesc, input, outputPrefix, v_inputFiles, saveOriginInput, params);
        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Create input data failed\n",
                   GetTime().c_str());
            DestroyResource(modelId, modelDesc, v_context, params);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Create input data success\n",
                   GetTime().c_str());
        }

        // 动态shape适配
        if (params['v'] != "") {
            if (params['w'] != "") {
                aclTensorDesc *inputDesc[v_inputFiles.size()];
                ret = CreateInputDesc(input, v_inputFiles, inputDesc, params);
                if (ret != SUCCESS) {
                    printf("%s - E - [XACL]: Create input description failed\n",
                           GetTime().c_str());
                    DestroyResource(modelId, modelDesc, v_context, params);
                    return FAILED;
                } else {
                    printf("%s - I - [XACL]: Create input description success\n",
                           GetTime().c_str());
                }
            } else {
                printf("%s - E - [XACL]: When dynamicShape function is enable, must set dynamicOutSize\n",
                       GetTime().c_str());
                DestroyResource(modelId, modelDesc, v_context, params);
                return FAILED;
            }
        }

        // 创建输出
        aclmdlDataset *output = nullptr;
        ret = CreateOutput(modelDesc, output, input, params);
        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Create output data failed\n",
                   GetTime().c_str());
            DestroyResource(modelId, modelDesc, v_context, params);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Create output data success\n",
                   GetTime().c_str());
        }

        // 执行推理
        ret = Execute(input, output, modelDesc, modelId, v_context, totalSamplesTime, outputPrefix, saveOriginInput, 0, params);

        stepNumber++;

        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Execute acl model failed\n",
                   GetTime().c_str());
            DestroyResource(modelId, modelDesc, v_context, params);
            return FAILED;
        }
        DestroyResource(modelId, modelDesc, v_context, params);
    }

    ret = WriteResult(stepNumber, totalSamplesTime, params);

    if (ret != SUCCESS) {
        printf("%s - E - [XACL]: Write acl result failed\n",
               GetTime().c_str());
        return FAILED;
    }

    return SUCCESS;
}