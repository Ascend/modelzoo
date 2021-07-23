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

int THOUSAND = 1000;
int MILLION = 1000000;
size_t MALLOC_ONE = 1;

void Help() {
    printf("%s - I - [XACL]: Usage: ./xacl_fmk [input parameters]\n"
           "-m=model                 Required, om model file path\n"
           "                         Relative and absolute paths are supported\n"
           "-o=outputPath            Required, path of output files\n"
           "                         Relative and absolute paths are supported\n"
           "\n"
           "-i=inputFiles            Optional, input bin files or input file directories, use commas (,) to separate multiple inputs\n"
           "                         Relative and absolute paths are supported, set inputs to all zeros if not specified\n"
           "-g=mergeInput            Optional, whether merge input by batch size, only take effect in directories input\n"
           "                         The default value is false, in this case, each input must be saved in N batches\n"
           "                         Otherwise, each input must be saved in 1 batch and will be merged to N batches automatically\n"
           "-d=dumpJson              Optional, Configuration file used to save operator input and output data\n"
           "                         The default value is NULL, indicating that operator input and output data is not saved\n"
           "-n=nodeId                Optional, ID of the NPU used for inference\n"
           "                         The default value is 0, indicating that device 0 is used for inference\n"
           "-l=loopNum               Optional, The number of inference times\n"
           "                         The default value is 1, indicating that inference is performed once\n"
           "-b=batchSize             Optional, Size of the static batch\n"
           "                         The default value is 1, indicating that the static batch is 1\n"
           "                         Static batch will be disabled when dynamic batch has been set\n"
           "-v=dynamicShape          Optional, Size of the dynamic shape\n"
           "                         Use semicolon (;) to separate each input, use commas (,) to separate each dim\n"
           "                         The default value is NULL, indicating that the dynamicShape function is disabled\n"
           "                         Enter the actual shape size when the dynamicShape function is enabled\n"
           "-w=dynamicSize           Optional, Size of the output memory\n"
           "                         Use semicolon (;) to separate each output\n"
           "                         The default value is NULL, indicating that the dynamicShape function is disabled\n"
           "                         Enter the actual output size when the dynamicShape function is enabled\n"
           "-x=imageRank             Optional, Size of the height and width rank, use commas (,) to separate\n"
           "                         The default value is NULL, indicating that the image rank function is disabled\n"
           "                         Enter the actual height and width size when the image rank function is enabled\n"
           "-y=batchRank             Optional, Size of the batch rank, cannot be used with heightRank or widthRank\n"
           "                         The default value is 0, indicating that the batch rank function is disabled\n"
           "                         Enter the actual size of the batch when the batch rank function is enabled\n"
           "-z=dimsRank              Optional, Size of the dims rank, use commas (,) to separate\n"
           "                         The default value is NULL, indicating that the dims rank function is disabled\n"
           "                         Enter the actual size of each dims when the dims rank function is enabled\n"
           "-r=remoteDevice          Optional, Whether the NPU is deployed remotely\n"
           "                         The default value is 0, indicating that the NPU is co-deployed as 1951DC\n"
           "                         The value 1 indicates that the NPU is deployed remotely as 1951MDC\n"
           "\n"
           "-h=help                  Show this help message\n", GetTime().c_str());
}

string GetTime() {
    struct timeval timeEval;
    gettimeofday(&timeEval, NULL);
    int milliSecond = timeEval.tv_usec / THOUSAND;

    time_t timeStamp;
    time(&timeStamp);
    char secondTime[20];
    strftime(secondTime, sizeof(secondTime), "%Y-%m-%d %H:%M:%S", localtime(&timeStamp));

    char milliTime[24];
    snprintf(milliTime, sizeof(milliTime), "%s.%03d", secondTime, milliSecond);

    return milliTime;
}

string GetInputName(string inputFile) {
    vector<string> v_inputNameBin;
    Split(inputFile, v_inputNameBin, "/");
    string inputNameBin = v_inputNameBin[v_inputNameBin.size() - 1];
    vector<string> v_inputName;
    Split(inputNameBin, v_inputName, ".");
    return v_inputName[0];
}

string FormatInt(string Idx, size_t formatSize) {
    size_t sizeIdx = Idx.size();
    if (sizeIdx < formatSize) {
        for (size_t i = 0; i < formatSize - sizeIdx; ++i) {
            Idx = "0" + Idx;
        }
    }
    return Idx;
}

void Split(const string &tokens, vector<string> &v_tokens, const string &delimiters = ";") {
    string::size_type lastPos = tokens.find_first_not_of(delimiters, 0);
    string::size_type pos = tokens.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        v_tokens.push_back(tokens.substr(lastPos, pos - lastPos));
        lastPos = tokens.find_first_not_of(delimiters, pos);
        pos = tokens.find_first_of(delimiters, lastPos);
    }
}

void ScanFiles(vector <string> &v_fileList, string inputDirectory) {
    const char *str = inputDirectory.c_str();
    DIR *dir = opendir(str);
    struct dirent *p = NULL;
    while ((p = readdir(dir)) != NULL) {
        if (p->d_name[0] != '.') {
            string name = string(p->d_name);
            v_fileList.push_back(inputDirectory + '/' + name);
        }
    }
    closedir(dir);

    if (v_fileList.size() == 0) {
        printf("%s - E - [XACL]: No file in the directory: %s\n",
               GetTime().c_str(), str);
    }
}

void MergeInputFile(vector <string> &fileList, string &outputFile) {
    ofstream fileOut(outputFile, ofstream::binary);
    for (uint32_t fileIdx = 0; fileIdx < fileList.size(); ++fileIdx) {
        ifstream binFile(fileList[fileIdx], ifstream::binary);

        binFile.seekg(0, binFile.beg);
        while (!binFile.eof()) {
            char szBuf[256] = {'\0'};
            binFile.read(szBuf, sizeof(char) * 256);
            int length = binFile.gcount();
            fileOut.write(szBuf, length);
        }
        binFile.close();
    }
    fileOut.close();
}

void DestroyModel(uint32_t modelId) {
    aclError ret;
    ret = aclmdlUnload(modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlUnload return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
}

void DestroyModelDesc(aclmdlDesc *modelDesc) {
    aclError ret;
    ret = aclmdlDestroyDesc(modelDesc);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlDestroyDesc return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
}

void DestroyModelContext(vector<aclrtContext> &v_context) {
    aclError ret;
    for (auto iter = v_context.begin(); iter != v_context.end(); iter++) {
        ret = aclrtDestroyContext(*iter);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtDestroyContext return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
    }
}

void DestroyDevice(uint32_t nodeId) {
    aclError ret;
    ret = aclrtResetDevice(nodeId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtResetDevice return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }

    printf("%s - I - [XACL]: Start to finalize acl, aclFinalize interface adds 2s delay to upload device logs\n",
           GetTime().c_str());
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclFinalize return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }

    printf("%s - I - [XACL]: Finalize acl success\n",
           GetTime().c_str());
}

void DestroyDataset(aclmdlDataset *dataset) {
    aclError ret;
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        void *data = aclGetDataBufferAddr(dataBuffer);

        ret = aclrtFree(data);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclDestroyDataBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
    }

    ret = aclmdlDestroyDataset(dataset);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlDestroyDataset return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
    printf("%s - I - [XACL]: Destroy input data success\n",
           GetTime().c_str());
}

void *ReadBinFile(string fileName, uint32_t &fileSize, uint32_t remoteFlag) {
    ifstream binFile(fileName, ifstream::binary);
    if (!binFile.is_open()) {
        printf("%s - E - [XACL]: Open input file failed\n",
               GetTime().c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    binFile.seekg(0, binFile.beg);

    aclError ret;
    void *binFileBufferData = nullptr;
    if (remoteFlag == 0) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            binFile.close();
            return nullptr;
        }
    } else {
        ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            binFile.close();
            return nullptr;
        }
    }

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;

    return binFileBufferData;
}

void *GetDeviceBufferFromFile(string fileName, uint32_t &fileSize, uint32_t remoteFlag) {
    uint32_t inputHostBuffSize = 0;
    void *inputHostBuff = ReadBinFile(fileName, inputHostBuffSize, remoteFlag);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }

    if (remoteFlag == 0) {
        aclError ret;
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        ret = aclrtMalloc(&inBufferDev, inBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ret = aclrtFreeHost(inputHostBuff);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());

            ret = aclrtFree(inBufferDev);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            ret = aclrtFreeHost(inputHostBuff);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            return nullptr;
        }
        ret = aclrtFreeHost(inputHostBuff);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
        fileSize = inBufferSize;
        return inBufferDev;
    } else {
        fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}

Result InitAndCheckParams(int argc, char *argv[], map<char, string> &params) {
    // 设置默认值
    params['m'] = "";
    params['i'] = "";
    params['o'] = "./";
    params['d'] = "";
    params['v'] = "";
    params['w'] = "";
    params['x'] = "";
    params['y'] = "";
    params['z'] = "";
    params['n'] = "0";
    params['l'] = "1";
    params['b'] = "1";
    params['r'] = "0";
    params['g'] = "0";
    // 获取入参
    while (1) {
        int optionIdx = 0;
        int optionInput;
        struct option OPTIONS[] = {
                {"modelFile",    1, 0, 'm'},

                {"inputFiles",   1, 0, 'i'},
                {"outputPath",   1, 0, 'o'},
                {"dumpJson",     1, 0, 'd'},

                {"dynamicShape", 1, 0, 'v'},
                {"dynamicSize",  1, 0, 'w'},
                {"imageRank",    1, 0, 'x'},
                {"batchRank",    1, 0, 'y'},
                {"dimsRank",     1, 0, 'z'},

                {"nodeId",       1, 0, 'n'},
                {"loopNum",      1, 0, 'l'},
                {"batchSize",    1, 0, 'b'},
                {"remoteFlag",   1, 0, 'r'},
                {"mergeFlag",    1, 0, 'g'},

                {"help",         0, 0, 'h'},
        };

        optionInput = getopt_long(argc, argv, "i:g:o:m:n:l:d:b:v:w:x:y:z:r:h", OPTIONS, &optionIdx);
        if (optionInput == -1) {
            break;
        }
        switch (optionInput) {
            case 'm': {
                params['m'] = string(optarg);
                printf("%s - I - [XACL]: NPU model file: %s\n",
                       GetTime().c_str(), params['m'].c_str());
                break;
            }

            case 'i': {
                params['i'] = string(optarg);
                printf("%s - I - [XACL]: Input files: %s\n",
                       GetTime().c_str(), params['i'].c_str());
                break;
            }

            case 'o': {
                params['o'] = string(optarg);
                printf("%s - I - [XACL]: Output file path: %s\n",
                       GetTime().c_str(), params['o'].c_str());
                break;
            }

            case 'd': {
                params['d'] = string(optarg);
                printf("%s - I - [XACL]: Dump config file: %s\n",
                       GetTime().c_str(), params['d'].c_str());
                break;
            }

            case 'v': {
                params['v'] = string(optarg);
                printf("%s - I - [XACL]: Dynamic shape size: %s\n",
                       GetTime().c_str(), params['v'].c_str());
                break;
            }

            case 'w': {
                params['w'] = string(optarg);
                printf("%s - I - [XACL]: Dynamic output memory size: %s\n",
                       GetTime().c_str(), params['w'].c_str());
                break;
            }

            case 'x': {
                params['x'] = string(optarg);
                printf("%s - I - [XACL]: Image size rank: %s\n",
                       GetTime().c_str(), params['x'].c_str());
                break;
            }

            case 'y': {
                params['y'] = string(optarg);
                printf("%s - I - [XACL]: Batch size rank: %s\n",
                       GetTime().c_str(), params['y'].c_str());
                break;
            }

            case 'z': {
                params['z'] = string(optarg);
                printf("%s - I - [XACL]: Dims size rank: %s\n",
                       GetTime().c_str(), params['z'].c_str());
                break;
            }

            case 'n': {
                params['n'] = string(optarg);
                printf("%s - I - [XACL]: NPU device index: %s\n",
                       GetTime().c_str(), params['n'].c_str());
                break;
            }

            case 'l': {
                params['l'] = string(optarg);
                printf("%s - I - [XACL]: Execution loops: %s\n",
                       GetTime().c_str(), params['l'].c_str());
                break;
            }

            case 'b': {
                params['b'] = string(optarg);
                printf("%s - I - [XACL]: Static batch size: %s\n",
                       GetTime().c_str(), params['b'].c_str());
                break;
            }

            case 'r': {
                params['r'] = string(optarg);
                printf("%s - I - [XACL]: Remote device flag: %s\n",
                       GetTime().c_str(), params['r'].c_str());
                break;
            }

            case 'g': {
                params['g'] = string(optarg);
                printf("%s - I - [XACL]: Merge input flag: %s\n",
                       GetTime().c_str(), params['g'].c_str());
                break;
            }

            case 'h': {
                Help();
                return FAILED;
            }
        }
    }
    // 判断模型文件是否存在，不存在则报错退出
    if (params['m'] == "") {
        printf("%s - E - [XACL]: NPU model file (-m) parameter is required\n",
               GetTime().c_str());
        return FAILED;
    }

    // 判断output目录是否存在，若不存在则创建目录，创建失败则退出
    if (CreateDir(params['o']) != SUCCESS) {
        return FAILED;
    }

    // Image、batch和dims分档以及动态Shape不能同时存在
    if ((params['y'] != "" && params['x'] != "") ||
        (params['y'] != "" && params['z'] != "") ||
        (params['y'] != "" && params['v'] != "") ||
        (params['x'] != "" && params['z'] != "") ||
        (params['x'] != "" && params['v'] != "") ||
        (params['z'] != "" && params['v'] != "")) {
        printf("%s - E - [XACL]: Can't set batch, image, dims rank size or dynamic shape at the same time\n",
               GetTime().c_str());
        return FAILED;
    }

    // batch分档时将实际batch赋值为batch分档大小
    if (params['y'] != "") {
        printf("%s - I - [XACL]: Due to batch rank size is set, static batch size resets to the same value\n",
               GetTime().c_str());
        params['b'] = params['y'];
        printf("%s - I - [XACL]: Static batch size resets to %s\n",
               GetTime().c_str(), params['b'].c_str());
    }
    return SUCCESS;
}

Result CreateDir(string &pathName) {
    string tempPath = "";
    vector<string> v_pathName;
    if (pathName.find("./") == 0) {
        tempPath = "./";
    } else if (pathName.find("/") == 0) {
        tempPath = "/";
    }

    Split(pathName, v_pathName, "/");

    for (uint32_t i = 0; i < v_pathName.size(); i++) {
        if (v_pathName[i] == ".") continue;
        tempPath += v_pathName[i] + "/";
        if (access(tempPath.c_str(), F_OK) == -1) {
            printf("%s - I - [XACL]: Path %s is not exist, try to create it\n",
                   GetTime().c_str(), tempPath.c_str());
            if (mkdir(tempPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
                printf("%s - E - [XACL]: Create path failed\n",
                       GetTime().c_str());
                return FAILED;
            }
        }
    }
    pathName = tempPath;
    return SUCCESS;
}

Result InitResource(vector<aclrtContext> &v_context, map<char, string> &params) {
    aclError ret;
    ret = aclInit((char *) params['d'].c_str());
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclInit return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }

    ret = aclrtSetDevice(atoi(params['n'].c_str()));
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtSetDevice return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }

    aclrtContext context;
    ret = aclrtCreateContext(&context, atoi(params['n'].c_str()));
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtCreateContext return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }
    v_context.push_back(context);

    return SUCCESS;
}

Result DestroyResource(uint32_t modelId, aclmdlDesc *modelDesc, vector<aclrtContext> &v_context, map<char, string> &params) {
    if (modelDesc && modelId) {
        DestroyModel(modelId);
        DestroyModelDesc(modelDesc);
    }
    DestroyModelContext(v_context);
    DestroyDevice(atoi(params['n'].c_str()));
    return SUCCESS;
}

Result LoadModelFromFile(uint32_t * modelId, map<char, string> & params) {
    aclError ret;
    ret = aclmdlLoadFromFile(params['m'].c_str(), modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlLoadFromFile return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }
    return SUCCESS;
}

Result CreateModelDesc(aclmdlDesc *&modelDesc, uint32_t modelId) {
    aclError ret;
    modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDesc return failed\n",
               GetTime().c_str());
        return FAILED;
    }
    ret = aclmdlGetDesc(modelDesc, modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlGetDesc return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }
    return SUCCESS;
}

Result CreateInputDesc(aclmdlDataset *input, vector<string> v_inputFiles, aclTensorDesc *inputDesc[], map<char, string> &params) {
    vector<string> v_dynamicShape;
    Split(params['v'], v_dynamicShape, ";");
    if (v_dynamicShape.size() != v_inputFiles.size()) {
        printf("%s - I - [XACL]: dynamicShape input numbers %zu are not equal with actually input numbers %zu\n",
               GetTime().c_str(), v_dynamicShape.size(), v_inputFiles.size());
        DestroyDataset(input);
        return FAILED;
    }

    for (uint32_t inIdx = 0; inIdx < v_dynamicShape.size(); ++inIdx) {
        aclError ret;
        vector<string> v_shapes;
        Split(v_dynamicShape[inIdx], v_shapes, ",");
        int64_t shapes[v_shapes.size()];
        for (uint32_t dimIdx = 0; dimIdx < v_shapes.size(); ++dimIdx) {
            printf("%s - I - [XACL]: The %d input dynamicShape index %d is: %s\n",
                   GetTime().c_str(), inIdx, dimIdx, v_shapes[dimIdx].c_str());
            shapes[dimIdx] = atoi(v_shapes[dimIdx].c_str());
        }
        //ACL_FLOAT16 与 ACL_FORMAT_NHWC随意填写，暂时不生效
        inputDesc[inIdx] = aclCreateTensorDesc(ACL_FLOAT16, v_shapes.size(), shapes, ACL_FORMAT_NHWC);

        ret = aclmdlSetDatasetTensorDesc(input, inputDesc[inIdx], inIdx);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - I - [XACL]: Interface of aclmdlSetDatasetTensorDesc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            DestroyDataset(input);
            return FAILED;
        }
    }
    return SUCCESS;
}

Result CreateInput(aclmdlDesc *modelDesc, aclmdlDataset *&input, string &outputPrefix, vector<string> v_inputFiles, vector<vector<string>> saveOriginInput, map<char, string> &params) {
    input = aclmdlCreateDataset();
    if (input == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDataset return failed\n",
               GetTime().c_str());
        return FAILED;
    }

    size_t inputSize = aclmdlGetNumInputs(modelDesc);

    if (v_inputFiles.empty()) {
        // 不指定输入，创建全零输入
        printf("%s - I - [XACL]: Input type is empty, create all zero inputs\n",
               GetTime().c_str());
        outputPrefix = params['o'] + "all_zero_input";
        printf("%s - I - [XACL]: The number of inputs queried through aclmdlGetNumInputs is: %zu\n",
               GetTime().c_str(), inputSize);
        for (size_t i = 0; i < inputSize; i++) {
            aclError ret;
            size_t bufferSizeZero = aclmdlGetInputSizeByIndex(modelDesc, i);
            printf("%s - I - [XACL]: The buffer size of input %zu: %zu\n",
                   GetTime().c_str(), i, bufferSizeZero);

            void *inputBuffer = nullptr;
            if (!atoi(params['r'].c_str())) {
                void *binFileBufferData = nullptr;
                ret = aclrtMallocHost(&binFileBufferData, bufferSizeZero);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    return FAILED;
                }

                memset(binFileBufferData, 0, bufferSizeZero);

                ret = aclrtMalloc(&inputBuffer, bufferSizeZero, ACL_MEM_MALLOC_NORMAL_ONLY);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                           GetTime().c_str(), bufferSizeZero, aclGetRecentErrMsg());
                    return FAILED;
                }

                ret = aclrtMemcpy(inputBuffer, bufferSizeZero, binFileBufferData, bufferSizeZero,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    return FAILED;
                }

                ret = aclrtFreeHost(binFileBufferData);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                }

            } else {
                ret = aclrtMalloc(&inputBuffer, bufferSizeZero, ACL_MEM_MALLOC_NORMAL_ONLY);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                           GetTime().c_str(), bufferSizeZero, aclGetRecentErrMsg());
                    return FAILED;
                }
                memset(inputBuffer, 0, bufferSizeZero);
            }

            aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer, bufferSizeZero);
            if (inputData == nullptr) {
                printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                       GetTime().c_str());
                DestroyDataset(input);
                return FAILED;
            }
            ret = aclmdlAddDatasetBuffer(input, inputData);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                return FAILED;
            }
        }
    } else {
        // 输入是指定bin文件
        printf("%s - I - [XACL]: Input type is bin file\n",
               GetTime().c_str());
        if (inputSize != v_inputFiles.size()) {
            printf("%s - E - [XACL]: Input file number not match, [%zu / %zu]\n",
                   GetTime().c_str(), v_inputFiles.size(), inputSize);
            return FAILED;
        }
        // 目录输入下，拼接最终输出前缀为 outputPath/1st_input_filename.bin
        if ((params['g'] != "0") && (!saveOriginInput.empty())){
            outputPrefix = params['o'];
        } else {
            outputPrefix = params['o'] + GetInputName(v_inputFiles[0]);
        }

        vector<void *> inputBuffer(v_inputFiles.size(), nullptr);
        for (size_t inIdx = 0; inIdx < inputSize; ++inIdx) {
            uint32_t bufferSize;
            inputBuffer[inIdx] = GetDeviceBufferFromFile(v_inputFiles[inIdx], bufferSize, atoi(params['r'].c_str()));
            aclDataBuffer *inputData = aclCreateDataBuffer((void *) (inputBuffer[inIdx]), bufferSize);
            if (inputData == nullptr) {
                printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                       GetTime().c_str());
                DestroyDataset(input);
                return FAILED;
            }

            aclError ret;
            ret = aclmdlAddDatasetBuffer(input, inputData);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                return FAILED;
            }
            printf("%s - I - [XACL]: The buffer size of input %zu: %d\n",
                   GetTime().c_str(), inIdx, bufferSize);
        }
    }

    return SUCCESS;
}

Result CreateOutput(aclmdlDesc *modelDesc, aclmdlDataset *&output, aclmdlDataset *input, map<char, string> &params) {
    if (modelDesc == nullptr) {
        printf("%s - E - [XACL]: No model desc, create output failed\n",
               GetTime().c_str());
        return FAILED;
    }

    output = aclmdlCreateDataset();
    if (output == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDataset return failed\n",
               GetTime().c_str());
        DestroyDataset(input);
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t outIdx = 0; outIdx < outputSize; ++outIdx) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, outIdx);
        if (bufferSize == 0) {
            if (params['w'] == "") {
                printf("%s - I - [XACL]: Output size is zero, malloc 1 byte\n",
                       GetTime().c_str());
                bufferSize = MALLOC_ONE;
            } else {
                vector<string> v_dynamicSize;
                Split(params['w'], v_dynamicSize, ";");
                bufferSize = atoi(v_dynamicSize[outIdx].c_str());
                printf("%s - I - [XACL]: Dynamic output, set size to %zu\n",
                       GetTime().c_str(), bufferSize);
            }
        }

        void *outputBuffer = nullptr;

        aclError ret;
        ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                   GetTime().c_str(), bufferSize, aclGetRecentErrMsg());
            DestroyDataset(input);
            DestroyDataset(output);
            return FAILED;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        if (outputData == nullptr) {
            printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                   GetTime().c_str());
            DestroyDataset(input);
            DestroyDataset(output);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output, outputData);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            DestroyDataset(input);
            DestroyDataset(output);
            return FAILED;
        }
    }
    return SUCCESS;
}

Result Execute(aclmdlDataset *input, aclmdlDataset *output, aclmdlDesc *modelDesc, uint32_t modelId, vector<aclrtContext> &v_context, float &totalSamplesTime, string outputPrefix, vector<vector<string>> saveOriginInput, uint32_t fileIdx, map<char, string> &params) {
    // 拆分 imageRank
    uint32_t heightRank = 0;
    uint32_t widthRank = 0;
    if (params['x'] != "") {
        vector<string> v_imageRank;
        Split(params['x'], v_imageRank, ",");
        for (uint32_t Idx = 0; Idx < v_imageRank.size(); ++Idx) {
            if (Idx == 0) {
                printf("%s - I - [XACL]: heightRank is: %s\n",
                       GetTime().c_str(), v_imageRank[Idx].c_str());
                heightRank = atoi(v_imageRank[Idx].c_str());
            } else if (Idx == 1) {
                printf("%s - I - [XACL]: widthRank is: %s\n",
                       GetTime().c_str(), v_imageRank[Idx].c_str());
                widthRank = atoi(v_imageRank[Idx].c_str());
            } else {
                printf("%s - E - [XACL]: imageRank only has two members which represent the height and width\n",
                       GetTime().c_str());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }
        }

        // 动态HW时，必须同时提供H和W大小
        if ((heightRank != 0 && widthRank == 0) || (heightRank == 0 && widthRank != 0)) {
            printf("%s - E - [XACL]: Both height and width are needed when enable imageRank\n",
                   GetTime().c_str());
            DestroyDataset(input);
            DestroyDataset(output);
            return FAILED;
        }
    }

    // 拆分 dimsRank
    uint32_t dimCount = 0;
    aclmdlIODims currentDims;
    if (params['z'] != "") {
        vector<string> v_dimsRank;
        Split(params['z'], v_dimsRank, ",");
        for (uint32_t Idx = 0; Idx < v_dimsRank.size(); ++Idx) {
            printf("%s - I - [XACL]: Dims rank index %d is: %s\n",
                   GetTime().c_str(), Idx, v_dimsRank[Idx].c_str());
            currentDims.dims[Idx] = atoi(v_dimsRank[Idx].c_str());
            dimCount++;
        }
        printf("%s - I - [XACL]: Dims rank size is: %d\n",
               GetTime().c_str(), dimCount);
        currentDims.dimCount = dimCount;
    }

    struct timeval startTimeStamp;
    struct timeval endTimeStamp;
    float totalTime = 0;
    float costTime;
    float startTime;
    float endTime;
    for (uint32_t loopIdx = 0; loopIdx < atoi(params['l'].c_str()); ++loopIdx) {
        aclError ret;

        if (atoi(params['y'].c_str()) > 0) {
            size_t rankIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &rankIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }

            ret = aclmdlSetDynamicBatchSize(modelId, input, rankIndex, atoi(params['y'].c_str()));
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetDynamicBatchSize return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }
        }

        if (heightRank > 0 && widthRank > 0) {
            size_t rankIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &rankIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }

            ret = aclmdlSetDynamicHWSize(modelId, input, rankIndex, heightRank, widthRank);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetDynamicHWSize return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }
        }

        if (dimCount > 0) {
            size_t rankIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &rankIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }

            ret = aclmdlSetInputDynamicDims(modelId, input, rankIndex, &currentDims);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetInputDynamicDims return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                DestroyDataset(input);
                DestroyDataset(output);
                return FAILED;
            }
        }

        gettimeofday(&startTimeStamp, NULL);

        ret = aclmdlExecute(modelId, input, output);

        gettimeofday(&endTimeStamp, NULL);

        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlExecute return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());

            printf("%s - E - [XACL]: Run acl model failed\n",
                   GetTime().c_str());
            DestroyDataset(input);
            DestroyDataset(output);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Run acl model success\n",
                   GetTime().c_str());
        }

        costTime = (1.0 * (endTimeStamp.tv_sec - startTimeStamp.tv_sec) * MILLION +
                      (endTimeStamp.tv_usec - startTimeStamp.tv_usec)) / THOUSAND;
        startTime = (1.0 * startTimeStamp.tv_sec * MILLION + startTimeStamp.tv_usec) / THOUSAND;
        endTime = (1.0 * endTimeStamp.tv_sec * MILLION + endTimeStamp.tv_usec) / THOUSAND;
        totalTime += costTime;
        printf("%s - I - [XACL]: Loop %d, start timestamp %4.0f, end timestamp %4.0f, cost time %4.2fms\n",
               GetTime().c_str(), loopIdx, startTime, endTime, costTime);

        string outputBinFileName(outputPrefix + "_output_");
        string loopName = "_" + FormatInt(to_string(loopIdx), 3);
        for (size_t outIndex = 0; outIndex < aclmdlGetDatasetNumBuffers(output); ++outIndex) {
            string outputBinFileNameIdx = outputBinFileName + FormatInt(to_string(outIndex), 2) + loopName + ".bin";
            FILE *fop = fopen(outputBinFileNameIdx.c_str(), "wb+");
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, outIndex);
            void *data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
            if (atoi(params['r'].c_str()) == 0) {
                void *outHostData = NULL;
                ret = aclrtMallocHost(&outHostData, len);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }
                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }

                size_t len1 = fwrite(outHostData, sizeof(char), len, fop);
                if (len1 != len) {
                    printf("%s - E - [XACL]: Write output bin file failed\n",
                           GetTime().c_str());
                }
                fclose(fop);

                ret = aclrtFreeHost(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }
            } else {
                void *outHostData = NULL;
                ret = aclrtMalloc(&outHostData, len, ACL_MEM_MALLOC_NORMAL_ONLY);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }

                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_DEVICE);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }

                size_t len1 = fwrite(outHostData, sizeof(char), len, fop);
                if (len1 != len) {
                    printf("%s - E - [XACL]: Dump output to file return failed\n",
                           GetTime().c_str());
                }
                fclose(fop);

                ret = aclrtFree(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    DestroyDataset(input);
                    DestroyDataset(output);
                    return FAILED;
                }
            }

            if ((params['g'] != "0") && (!saveOriginInput.empty())) {
                ifstream binFile(outputBinFileNameIdx, ifstream::binary);
                binFile.seekg(0, binFile.end);
                uint32_t fileLength = binFile.tellg();
                binFile.seekg(0, binFile.beg);

                uint32_t binFileBufferLen = fileLength / atoi(params['b'].c_str());
                for(int sampleIdx = 0; sampleIdx < atoi(params['b'].c_str()); sampleIdx++){
                    char *dataBuf = new char[binFileBufferLen];
                    string outputOriginName = outputPrefix + GetInputName(saveOriginInput[fileIdx][sampleIdx]) + "_output_";
                    string outputOriginNameIdx = outputOriginName + FormatInt(to_string(outIndex), 2) + loopName + ".bin";
                    ofstream fileOut(outputOriginNameIdx, ofstream::binary);
                    binFile.read(dataBuf, binFileBufferLen*sizeof(char));
                    fileOut.write(dataBuf, binFileBufferLen*sizeof(char));
                    fileOut.close();
                    delete [] dataBuf;
                }
                binFile.close();
                remove(outputBinFileNameIdx.c_str());
            }

            printf("%s - I - [XACL]: Dump output %ld to file success\n",
                   GetTime().c_str(), outIndex);
        }
    }

    totalSamplesTime = totalSamplesTime + (totalTime / atoi(params['l'].c_str()));

    printf("%s - I - [XACL]: Single step average NPU inference time of %d loops: %f ms %4.2f fps\n",
           GetTime().c_str(),
           atoi(params['l'].c_str()),
           (totalTime / atoi(params['l'].c_str())),
           (THOUSAND * atoi(params['l'].c_str()) * atoi(params['b'].c_str()) / totalTime));

    DestroyDataset(input);
    DestroyDataset(output);

    return SUCCESS;
}

Result WriteResult(uint32_t stepNumber, float totalSamplesTime, map<char, string> &params) {

    uint32_t samplesNumber = stepNumber * atoi(params['b'].c_str());
    float totalAverageTime = totalSamplesTime / stepNumber;
    float totalAverageFPS = THOUSAND * samplesNumber / totalSamplesTime;

    printf("%s - I - [XACL]: %d samples average NPU inference time of %d steps: %f ms %4.2f fps\n",
           GetTime().c_str(), samplesNumber, stepNumber, totalAverageTime, totalAverageFPS);

    vector<string> v_modelFile;
    Split(params['m'], v_modelFile, "/");
    string modelName = v_modelFile[v_modelFile.size() - 1];

    vector<string> v_modelName;
    Split(modelName, v_modelName, ".");
    string resultFileName = params['o'] + v_modelName[0] + "_performance.txt";

    ofstream resultFile(resultFileName.c_str(), ofstream::out);
    if (!resultFile) {
        printf("%s - I - [XACL]: Open acl result file failed\n",
               GetTime().c_str());
        return FAILED;
    } else {
        printf("%s - I - [XACL]: Write acl result to file %s\n",
               GetTime().c_str(), resultFileName.c_str());
    }

    resultFile << samplesNumber << " samples average NPU inference time of " << stepNumber << " steps: ";
    resultFile << totalAverageTime << "ms " << totalAverageFPS << " fps" << endl;
    resultFile.close();

    return SUCCESS;
}
