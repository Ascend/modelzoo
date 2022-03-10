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
                std::cout << "[INFO]om = " << config.om << std::endl;
                break;
            case 'b':
                config.dataDir    = std::string(optarg);
                std::cout << "[INFO]dataDir = " << config.dataDir << std::endl;
                break;
            case 'c':
                config.outDir    = std::string(optarg);
                std::cout << "[INFO]outDir = " << config.outDir << std::endl;
                break;
            case 'd':
                config.batchSize = atoi(optarg);
                std::cout << "[INFO]batchSize = " << config.batchSize << std::endl;
                break;
            case 'e':
                config.deviceId = atoi(optarg);
                std::cout << "[INFO]deviceId = " << config.deviceId << std::endl;
                break;
            case 'f':
                config.loopNum    = atoi(optarg);
                std::cout << "[INFO]loopNum = " << config.loopNum << std::endl;
                break;
            case 'g':
                config.modelType = std::string(optarg);
                std::cout << "[INFO]modelType = " << config.modelType << std::endl;
                break;
            case 'h':
                config.imgType = std::string(optarg);
                std::cout << "[INFO]imgType = " << config.imgType << std::endl;
                break;
            case 'i':
                config.framework = std::string(optarg);
                std::cout << "[INFO]framework = " << config.framework << std::endl;
                break;
            case 'j':
                config.useDvpp = atoi(optarg);
                std::cout << "[INFO]useDvpp = " << config.useDvpp << std::endl;
                break; 
            default:
                break;
        }
    }
    
}

aclError ParseParams(int argc, char** argv, Config& config, std::string& errorMsg)
{
    getCommandLineParam(argc, argv, config);
    if (config.om.empty() || !FileExists(config.om)) {
        //std::cout << "om is empty" << std::endl;
        errorMsg = "om path is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    
    if (config.dataDir.empty() || !FolderExists(config.dataDir)) {
        errorMsg = "data Dir is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    std::cout << "dataDir is " << config.dataDir << std::endl;


    if (!config.outDir.empty() && !FolderExists(config.outDir)) {
        mkdir(config.outDir.c_str(), 0755);
        std::cout << "outDir " << config.outDir << std::endl;
    }
    
    if(config.batchSize <= 0){
        errorMsg = "batch Size should be > 0";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    std::cout << "batchSize " << config.batchSize << std::endl;
    
    if (config.modelType.empty())
    {
        std::cout << "FLAGS_modelType is empty" << std::endl;
        errorMsg = "modelType is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    std::cout << "modelType " << config.modelType << std::endl;

    if (config.imgType.empty())
    {
        std::cout << "imgType is empty" << std::endl;
        errorMsg = "imgType is invalid";
        return ACL_ERROR_PARSE_PARAM_FAILED;
    }
    std::cout << "imgType " << config.imgType << std::endl;
    std::cout << "useDvpp is " << config.useDvpp << std::endl;
    std::cout << "parase params done" << std::endl;

    return ACL_ERROR_NONE;
}

aclError Process()
{	
    std::vector<std::string> fileNames;
    ret = GetFiles(cfg.dataDir, fileNames);
    CHECK_RET(ret);
    size_t fileNum = fileNames.size();
    std::cout << "************fileNum: " << fileNames.size() << std::endl;
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
        std::cout << "loopCnt " << loopCnt << " loopNum " << cfg.loopNum << std::endl;
        for(size_t i = 0; i< fileNum/cfg.batchSize; i++)
        {
            gettimeofday(&startTmp, NULL);
            std::vector<std::string> batchFileNames;
            for (int j = 0; j < cfg.batchSize; j++) {
                batchFileNames.push_back(fileNames[i*cfg.batchSize+j]);
            }

            if(cfg.useDvpp){
                ret = DvppInitInput(batchFileNames);
                if (ret !=0)
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
                if (ret !=0)
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
    std::cout << "model input num " << inputNum << std::endl;
    snprintf(tmpChr, sizeof(tmpChr), "model input num %zd\n", inputNum);
    outFile << tmpChr;
    
    cfg.inputNum = inputNum;
    for (size_t i = 0; i < inputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++)
    {
        size_t size = aclmdlGetInputSizeByIndex(cfg.modelDesc, i);
        cfg.inputInfo[i].size = size;
        std::cout << "model input[" << i << "] size " << cfg.inputInfo[i].size << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] size %zd\n",  i, cfg.inputInfo[i].size);
        outFile << tmpChr;
        
        aclmdlIODims dims;
        ret = aclmdlGetInputDims(cfg.modelDesc, i, &dims);
        CHECK_ACL_RET("aclmdlGetInputDims failed", ret)
    
        cfg.inputInfo[i].dimCount = dims.dimCount;
        ret = aclrtMemcpy(cfg.inputInfo[i].dims , cfg.inputInfo[i].dimCount * sizeof(int64_t), dims.dims, cfg.inputInfo[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        CHECK_ACL_RET("aclrtMemcpy failed", ret)       
        
        std::cout << "model input[" << i << "] dimCount " << cfg.inputInfo[i].dimCount << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] dimCount %zd\n", i, cfg.inputInfo[i].dimCount);
        outFile << tmpChr;
        for (size_t dimIdx = 0; dimIdx < cfg.inputInfo[i].dimCount; dimIdx++)
        {
            std::cout << "model input[" << i << "] dim[" << dimIdx << "] info " << cfg.inputInfo[i].dims[dimIdx] << std::endl;
            snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.inputInfo[i].dims[dimIdx]);
            outFile << tmpChr;
        }
        
        cfg.inputInfo[i].Format = aclmdlGetInputFormat(cfg.modelDesc, i);

        cfg.inputInfo[i].Type = aclmdlGetInputDataType(cfg.modelDesc, i);
        
        std::cout << "model input[" << i << "] format " << cfg.inputInfo[i].Format << " inputType " << cfg.inputInfo[i].Type << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] format %d inputType %d\n", i, cfg.inputInfo[i].Format, cfg.inputInfo[i].Type);
        outFile << tmpChr;
        
    
        //const char tmp[ACL_MAX_TENSOR_NAME_LEN] = aclmdlGetInputNameByIndex(cfg.modelDesc, i);
        cfg.inputInfo[i].Name = aclmdlGetInputNameByIndex(cfg.modelDesc, i);
        std::cout << "model input[" << i << "] name " << cfg.inputInfo[i].Name << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model input[%zd] name %s\n", i, cfg.inputInfo[i].Name);
        outFile << tmpChr;
        
        size_t index;
        ret = aclmdlGetInputIndexByName(cfg.modelDesc, cfg.inputInfo[i].Name, &index);
        CHECK_ACL_RET("aclmdlGetInputIndexByName failed", ret);
    
        if (i != index)
        {
            std::cout << "aclmdlGetInputNameByIndex not equal aclmdlGetInputIndexByName" << std::endl;
            return 1;
        }
    }

    //Get model output info
    size_t outputNum = aclmdlGetNumOutputs(cfg.modelDesc);
    std::cout << "model output num " << outputNum << std::endl;
    snprintf(tmpChr, sizeof(tmpChr), "model output num %zd\n", outputNum);
    outFile << tmpChr;
    
    cfg.outputNum = outputNum;
    for (size_t i = 0; i < outputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++)
    {
        size_t size = aclmdlGetOutputSizeByIndex(cfg.modelDesc, i);
        cfg.outputInfo[i].size = size;
        std::cout << "model output[" << i << "] size " << cfg.outputInfo[i].size << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] size %zd\n",  i, cfg.outputInfo[i].size);
        outFile << tmpChr;
    
        aclmdlIODims dims;
        ret = aclmdlGetOutputDims(cfg.modelDesc, i, &dims);
        CHECK_ACL_RET("aclmdlGetOutputDims failed", ret);

        cfg.outputInfo[i].dimCount = dims.dimCount;   
        ret = aclrtMemcpy(cfg.outputInfo[i].dims, cfg.outputInfo[i].dimCount * sizeof(int64_t), dims.dims, cfg.outputInfo[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        CHECK_ACL_RET("aclrtMemcpy failed", ret);
        
        std::cout << "model output[" << i << "] dimCount " << cfg.outputInfo[i].dimCount << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] dimCount %zd\n", i, cfg.outputInfo[i].dimCount);
        outFile << tmpChr;
        
        for (size_t dimIdx = 0; dimIdx < cfg.outputInfo[i].dimCount; dimIdx++)
        {
            std::cout << "model output[" << i << "] dim[" << dimIdx << "] info " << cfg.outputInfo[i].dims[dimIdx] << std::endl;
            snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] dim[%zd] info %ld\n", i, dimIdx, cfg.outputInfo[i].dims[dimIdx]);
            outFile << tmpChr;
        }
        
        cfg.outputInfo[i].Format = aclmdlGetOutputFormat(cfg.modelDesc, i);
        cfg.outputInfo[i].Type = aclmdlGetOutputDataType(cfg.modelDesc, i);
        std::cout << "model output[" << i << "] format " << cfg.outputInfo[i].Format << " outputType " << cfg.outputInfo[i].Type << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] format %d outputType %d\n", i, cfg.outputInfo[i].Format, cfg.outputInfo[i].Type);
        outFile << tmpChr;
            
        cfg.outputInfo[i].Name = aclmdlGetOutputNameByIndex(cfg.modelDesc, i);
        std::cout << "model output[" << i << "] name " << cfg.outputInfo[i].Name << std::endl;
        snprintf(tmpChr, sizeof(tmpChr), "model output[%zd] name %s\n", i, cfg.outputInfo[i].Name);
        outFile << tmpChr;
        
        size_t index;
        ret = aclmdlGetOutputIndexByName(cfg.modelDesc, cfg.outputInfo[i].Name, &index);
        CHECK_ACL_RET("aclmdlGetOutputIndexByName failed", ret);

        if (i != index)
        {
            std::cout << "aclmdlGetOutputNameByIndex not equal aclmdlGetOutputIndexByName" << std::endl;
            return 1;
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
    std::cout << std::endl;
    	
    avgTime = 1.0*inferTime/processedCnt/cfg.batchSize/1000;
    avgPreTime = 1.0*dataProcTime/processedCnt/cfg.batchSize/1000;
    
    if (cfg.useDvpp){
        std::cout << std::endl;
        std::cout << "DVPP performance details:" << std::endl;
        std::cout << "#############################################" << std::endl;
        std::unordered_map<std::string, long long>::iterator iter;
        for (iter = dvppTime.begin(); iter != dvppTime.end(); iter++){
            std::cout << iter->first << " using avg time " << 1.0*iter->second/processedCnt/cfg.batchSize/1000 << " ms" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "performance summary:" << std::endl;
    std::cout << "#############################################" << std::endl;
    std::cout << "total " << processedCnt*cfg.batchSize << " imgs processed and batch size " << cfg.batchSize << std::endl;

    std::cout << "avg preprocess time " << avgPreTime << " ms, " << 1.0*1000/avgPreTime << " imgs/s" << std::endl;
    std::cout << "avg inference time " << avgTime << " ms, " << 1.0*1000/avgTime << " imgs/s " << std::endl;
    SaveResult();	

}
