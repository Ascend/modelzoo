/* *
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* */

#include "util.h"
#include "infer_engine.h"
#include "acl/acl_base.h"
#include <gflags/gflags.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include <string.h>
#include <map>
#include <fcntl.h>
#include <sstream>
#include <cerrno> 
#include <errno.h> 
#include <sys/errno.h>
#include <unordered_map>

#include <memory>
#include <fstream>

#include <sys/time.h>
#include <unistd.h>
#include <time.h> 
#include <dirent.h> 
#include <stdarg.h>
#include <getopt.h>



using namespace std;
using std::cout;
using std::endl;


Resnet50Result resnet50Res;
Config cfg;
aclError ret;
int processedCnt;
long long dataProcTime = 0;
long long inferTime = 0;
float avgTime = 0;
float avgPreTime = 0;

extern std::unordered_map<std::string,long long> dvppTime;
extern DataFrame outputDataframe;

void getCommandLineParam(int argc, char** argv, Config& config)
{
    while (1)
    {
        int option_index = 0;
        struct option long_options[] =
        {
            {"om", 1, 0, 'a'},
            {"dataDir", 1, 0, 'b'},
            {"outDir", 1, 0, 'c'},
            {"batchSize", 1, 0, 'd'},
            {"deviceId", 1, 0, 'e'},
            {"loopNum", 1, 0, 'f'},
            {"modelType", 1, 0, 'g'},
            {"imgType", 1, 0, 'h'},
	        {"framework", 1, 0, 'i'},
            {"useDvpp", 1 , 0 , 'j'},
            {0, 0, 0, 0}
            
            
        };
        
        int c;
        c = getopt_long(argc, argv, "a:b:c:e:f:j:k:l:m:n:u:t:", long_options, &option_index);
        if (c == -1)
        {
            break;
        }
        
        switch (c)
        {
            case 'a':
                config.om = std::string(optarg);
                printf("[INFO]om = %s\n", config.om.c_str());
                break;
            case 'b':
                config.dataDir    = std::string(optarg);
                printf("[INFO]dataDir = %s\n", config.dataDir.c_str());
                break;
            case 'c':
                config.outDir    = std::string(optarg);
                printf("[INFO]outDir = %s\n", config.outDir.c_str());
                break;
            case 'd':
                config.batchSize = atoi(optarg);
                printf("[INFO]batchSize = %d\n", config.batchSize);
                break;
            case 'e':
                config.deviceId = atoi(optarg);
                printf("[INFO]deviceId = %d\n", config.deviceId);
                break;
            case 'f':
                config.loopNum    = atoi(optarg);
                printf("[INFO]loopNum = %d\n", config.loopNum);
                break;
            case 'g':
                config.modelType = std::string(optarg);
                printf("[INFO]modelType = %s\n", config.modelType.c_str());
                break;
            case 'h':
                config.imgType = std::string(optarg);
                printf("[INFO]imgType = %s\n", config.imgType.c_str());
                break;
            case 'i':
                config.framework = std::string(optarg);
                printf("[INFO]framework = %s\n", config.framework.c_str());
                break;
            case 'j':
                config.useDvpp = atoi(optarg);
                printf("[INFO]useDvpp = %d\n", config.useDvpp);
                break; 
            default:
                break;
        }
    }
    
}

// 只校验必须的参数
aclError ParseParams(int argc, char** argv, Config& config, std::string& errorMsg)
{
    getCommandLineParam(argc, argv, config);
    
    LOG("parase params start\n");

    if (config.om.empty() || !FileExists(config.om)) {
        LOG("om is empty\n");
        errorMsg = "om path is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    
    if (config.dataDir.empty() || !FolderExists(config.dataDir)) {
        errorMsg = "data Dir is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    LOG("dataDir %s \n", config.dataDir.c_str());


    if (!config.outDir.empty() && !FolderExists(config.outDir)) {
        LOG("output dir %s not exists, try to make dir.\n", config.outDir.c_str());
        mkdir(config.outDir.c_str(), 0755);
        LOG("outDir %s \n", config.outDir.c_str());
    }
    
    if(config.batchSize <= 0){
        errorMsg = "batch Size should be > 0";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    LOG("batchSize %zd \n", config.batchSize);
    
    if (config.modelType.empty())
    {
        LOG("FLAGS_modelType is empty\n");
        errorMsg = "modelType is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    LOG("modelType %s \n", config.modelType.c_str());

    if (config.imgType.empty())
    {
        LOG("imgType is empty\n");
        errorMsg = "imgType is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    LOG("imgType %s \n", config.imgType.c_str());
    
    LOG("useDvpp is %d \n", config.useDvpp);

    LOG("parase params done\n");
    return ACL_ERROR_NONE;
}

aclError Process()
{	
    std::vector<std::string> fileNames;
    ret = GetFiles(cfg.dataDir, fileNames);
    CHECK_RET(ret);
    size_t fileNum = fileNames.size();
    LOG("***********fileNum:%zd\n",fileNames.size());
    struct timeval startTmp, endTmp;
    
    //get img resize weight and height
    getImgResizeShape();
    
    if(cfg.useDvpp){
        ret = DvppSetup();
        CHECK_RET(ret);
    }

    size_t inferCnt = 0;
    size_t loopCnt = 0;
    while(loopCnt < cfg.loopNum)
    {
        LOG("loopCnt %d, loopNum %d\n", loopCnt, cfg.loopNum);
        for(size_t i = 0; i< fileNum/cfg.batchSize; i++)
        {
            gettimeofday(&startTmp, NULL);
            std::vector<std::string> batchFileNames;
            for (int j = 0; j < cfg.batchSize; j++) {
                batchFileNames.push_back(fileNames[i*cfg.batchSize+j]);
            }

            if(cfg.useDvpp){
                ret = DvppInitInput(batchFileNames);
                if (ret != 0)
                {
                    continue;
                }
            }	
            else{
                ret = InitInput(batchFileNames);
            }
            processedCnt++;
            gettimeofday(&endTmp, NULL);
            dataProcTime += (endTmp.tv_sec-startTmp.tv_sec)*1000000+(endTmp.tv_usec-startTmp.tv_usec);
            CHECK_RET(ret);
            
            ret = Inference();
            CHECK_RET(ret);
            
            ret = SaveBinPostprocess();
            CHECK_RET(ret);
        }

        //last img
        if (0 != fileNum % cfg.batchSize)
        {
            std::vector<std::string> batchFileNames;
            for(size_t i = (fileNum - fileNum % cfg.batchSize); i< fileNum; i++)
            {
                batchFileNames.push_back(fileNames[i]);
            }
            
            gettimeofday(&startTmp, NULL);

            if(cfg.useDvpp){
                ret = DvppInitInput(batchFileNames);
                if (ret != 0)
                {
                    continue;
                }
            }	
            else{
                ret = InitInput(batchFileNames);
            }
            processedCnt++;
            gettimeofday(&endTmp, NULL);
            dataProcTime += (endTmp.tv_sec-startTmp.tv_sec)*1000000+(endTmp.tv_usec-startTmp.tv_usec);
            CHECK_RET(ret);
            
            ret = Inference();
            CHECK_RET(ret);
            
            ret = SaveBinPostprocess();
            CHECK_RET(ret);   
        }
        loopCnt++;
    }
    return ACL_ERROR_NONE;
}

void SaveResult(){
    ofstream outfile("test_perform_static.txt");
    #if 0
    std::string model_name;
    int dex = (cfg.om).find_last_of("/");
    model_name = cfg.om.substr(dex+1);
    
    std:: string title = "model_name total batch top1 top5 pre_avg/ms pre_imgs/s infer_avg/ms infer_imgs/s mAP";
    outfile << title << endl;

    outfile << model_name << " ";        
    outfile << processedCnt*cfg.batchSize << " ";
    outfile << cfg.batchSize << " ";
    if(cfg.postprocessType == "resnet"){
        outfile << 1.0*resnet50Res.top1/resnet50Res.total << " " << 1.0*resnet50Res.top5/resnet50Res.total << " ";
    }
    else{

        outfile << "NA" << " " << "NA" << " "; 
    }

    outfile << avgPreTime << " " << 1.0*1000/avgPreTime << " ";
    outfile << avgTime << " " << 1.0*1000/avgTime << " ";
    outfile << endl;
    #endif
    char tmpCh[256];
    memset(tmpCh, 0, sizeof(tmpCh));
    snprintf(tmpCh, sizeof(tmpCh), "NN inference cost average time: %4.3f ms %4.3f fps/s\n", avgTime, (1.0 * 1000/avgTime));
    outfile << tmpCh;
    outfile.close();
		
}


aclError GetModelInputOutputInfo(Config& cfg)
{
    aclError ret;
    
    std::ofstream  outFile("modelInputOutputInfo", std::ios::trunc);
    char tmpChr[256] = {0};
    //Get model input info
    size_t inputNum = aclmdlGetNumInputs(cfg.modelDesc);
    LOG("model input num %zd\n", inputNum);
    snprintf(tmpChr, sizeof(tmpChr), "model input num %zd\n", inputNum);
    outFile << tmpChr;
    
    cfg.inputNum = inputNum;
    for (size_t i = 0; i < inputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++)
    {
        size_t size = aclmdlGetInputSizeByIndex(cfg.modelDesc, i);
        cfg.inputInfo[i].size = size;
        LOG("model input[%zd] size %zd\n", i, cfg.inputInfo[i].size);
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] size %zd\n",  i, cfg.inputInfo[i].size);
        outFile << tmpChr;
        
        aclmdlIODims dims;
        ret = aclmdlGetInputDims(cfg.modelDesc, i, &dims);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclmdlGetInputDims fail ret %d\n", ret);
            return 1;
        }
        
        cfg.inputInfo[i].dimCount = dims.dimCount;
        ret = aclrtMemcpy(cfg.inputInfo[i].dims , cfg.inputInfo[i].dimCount * sizeof(int64_t), dims.dims, cfg.inputInfo[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclrtMemcpy fail ret %d line %d\n", ret, __LINE__);
            return 1;
        }

        LOG("model input[%zd] dimCount %zd\n", i, cfg.inputInfo[i].dimCount);
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] dimCount %zd\n", i, cfg.inputInfo[i].dimCount);
        outFile << tmpChr;
        for (size_t dimIdx = 0; dimIdx < cfg.inputInfo[i].dimCount; dimIdx++)
        {
            LOG("model input[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.inputInfo[i].dims[dimIdx]);
            snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.inputInfo[i].dims[dimIdx]);
            outFile << tmpChr;
        }
        
        cfg.inputInfo[i].Format = aclmdlGetInputFormat(cfg.modelDesc, i);

        cfg.inputInfo[i].Type = aclmdlGetInputDataType(cfg.modelDesc, i);
        
        LOG("model input[%zd] format %d inputType %d\n", i, cfg.inputInfo[i].Format, cfg.inputInfo[i].Type);
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] format %d inputType %d\n", i, cfg.inputInfo[i].Format, cfg.inputInfo[i].Type);
        outFile << tmpChr;
        
    
        //const char tmp[ACL_MAX_TENSOR_NAME_LEN] = aclmdlGetInputNameByIndex(cfg.modelDesc, i);
        cfg.inputInfo[i].Name = aclmdlGetInputNameByIndex(cfg.modelDesc, i);
        LOG("model input[%zd] name %s\n", i, cfg.inputInfo[i].Name);
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] name %s\n", i, cfg.inputInfo[i].Name);
        outFile << tmpChr;
        
        size_t index;
        ret = aclmdlGetInputIndexByName(cfg.modelDesc, cfg.inputInfo[i].Name, &index);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclmdlGetInputIndexByName fail ret %d line %d\n", ret, __LINE__);
            return 1;
        }

        if (i != index)
        {
            LOG("aclmdlGetInputNameByIndex not equal aclmdlGetInputIndexByName\n");
            return 1;
        }
        else
        {
             LOG("model input name %s is belone to input %zd\n", cfg.inputInfo[i].Name, index);
        }

    }

    //Get model output info
    size_t outputNum = aclmdlGetNumOutputs(cfg.modelDesc);
    LOG("model output num %zd\n", outputNum);
    snprintf(tmpChr, sizeof(tmpChr), "model output num %zd\n", outputNum);
    outFile << tmpChr;
    
    cfg.outputNum = outputNum;
    for (size_t i = 0; i < outputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++)
    {
        size_t size = aclmdlGetOutputSizeByIndex(cfg.modelDesc, i);
        cfg.outputInfo[i].size = size;
        LOG("model output[%zd] size %zd\n", i, cfg.outputInfo[i].size);
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] size %zd\n",  i, cfg.outputInfo[i].size);
        outFile << tmpChr;
    
        aclmdlIODims dims;
        ret = aclmdlGetOutputDims(cfg.modelDesc, i, &dims);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclmdlGetOutputDims fail ret %d\n", ret);
            return 1;
        }
        
        cfg.outputInfo[i].dimCount = dims.dimCount;
        ret = aclrtMemcpy(cfg.outputInfo[i].dims, cfg.outputInfo[i].dimCount * sizeof(int64_t), dims.dims, cfg.outputInfo[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclrtMemcpy fail ret %d line %d\n", ret, __LINE__);
            return 1;
        }
        
        LOG("model output[%zd] dimCount %zd\n", i, cfg.outputInfo[i].dimCount);
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] dimCount %zd\n", i, cfg.outputInfo[i].dimCount);
        outFile << tmpChr;
        
        for (size_t dimIdx = 0; dimIdx < cfg.outputInfo[i].dimCount; dimIdx++)
        {
            LOG("model output[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.outputInfo[i].dims[dimIdx]);
            snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.outputInfo[i].dims[dimIdx]);
            outFile << tmpChr;
        }
        
        cfg.outputInfo[i].Format = aclmdlGetOutputFormat(cfg.modelDesc, i);
        cfg.outputInfo[i].Type = aclmdlGetOutputDataType(cfg.modelDesc, i);
        LOG("model output[%zd] format %d outputType %d\n", i, cfg.outputInfo[i].Format, cfg.outputInfo[i].Type);
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] format %d outputType %d\n", i, cfg.outputInfo[i].Format, cfg.outputInfo[i].Type);
        outFile << tmpChr;
            
        cfg.outputInfo[i].Name = aclmdlGetOutputNameByIndex(cfg.modelDesc, i);
        LOG("model output[%zd] name %s\n", i, cfg.outputInfo[i].Name);
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] name %s\n", i, cfg.outputInfo[i].Name);
        outFile << tmpChr;
        
        size_t index;
        ret = aclmdlGetOutputIndexByName(cfg.modelDesc, cfg.outputInfo[i].Name, &index);
        if (ACL_ERROR_NONE != ret)
        {
            LOG("aclmdlGetOutputIndexByName fail ret %d line %d\n", ret, __LINE__);
            return 1;
        }

        if (i != index)
        {
            LOG("aclmdlGetOutputNameByIndex not equal aclmdlGetOutputIndexByName\n");
            return 1;
        }
        else
        {
             LOG("model output name %s is belone to output %d\n", cfg.outputInfo[i].Name, index);
        }
        
    }    

    outFile.close();
    
    return ACL_ERROR_NONE;
    
}


int main(int argc, char** argv)
{
    processedCnt = 0;
    inferTime = 0;
   
    std::string errorMsg;
    ret = ParseParams(argc, argv, cfg, errorMsg);
    CHECK_ACL_RET(errorMsg, ret);
    
    ret = InitContext();
    CHECK_RET(ret);
    
    ret = LoadModel();
    CHECK_RET(ret);
    
    ret = GetModelInputOutputInfo(cfg);
    CHECK_RET(ret);

    ret = Process();
    CHECK_RET(ret);

    ret = UnloadModel();
    CHECK_RET(ret);

    ret = UnInitContext();
    CHECK_RET(ret);
    LOG("\n");
    	
    avgTime = 1.0*inferTime/processedCnt/cfg.batchSize/1000;
    avgPreTime = 1.0*dataProcTime/processedCnt/cfg.batchSize/1000;
    
    if (cfg.useDvpp){
        LOG("\n");
        LOG("DVPP performance details:\n");
        LOG("#############################################\n");
        std::unordered_map<std::string, long long>::iterator iter;
        for (iter = dvppTime.begin(); iter != dvppTime.end(); iter++){
            LOG("%s using avg time %0.2f ms\n",iter->first.c_str(),1.0*iter->second/processedCnt/cfg.batchSize/1000);
        }
        LOG("\n");
    }

    LOG("performance summary:\n");
    LOG("#############################################\n");
    LOG("total %ld imgs processed and batch size %ld\n", processedCnt*cfg.batchSize, cfg.batchSize);
    #if 0
    if(cfg.postprocessType == "resnet"){
        LOG("top1 ratio %0.3f top5 ratio %0.3f\n", 1.0*resnet50Res.top1/resnet50Res.total, 1.0*resnet50Res.top5/resnet50Res.total);
    }
    #endif

    LOG("avg preprocess time %0.2f ms, %0.2f imgs/s\n", avgPreTime, 1.0*1000/avgPreTime);
    LOG("avg inference time %0.2f ms, %0.2f imgs/s\n", avgTime, 1.0*1000/avgTime);

    SaveResult();	

}
