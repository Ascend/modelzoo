#include "acl/acl_mdl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include <vector>
#include <cstring>
#include <string>
#include <memory>
#include <fstream>
#include "stdio.h"
#include <sys/time.h>
#include <unistd.h>
#include <time.h> 
#include <dirent.h> 
#include <stdarg.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <libgen.h>
#include <string.h>
#include <getopt.h>
#include <map>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <cerrno> 
#include <errno.h> 
#include <sys/errno.h>
#include "ptest.h"
#include "common.h"
#include "file.h"

uint32_t deviceNum = 1;
uint32_t  modelId = 0;
aclmdlDesc* modelDesc = nullptr;
aclmdlDataset* modelInput = nullptr;
aclmdlDataset* modelOutput = nullptr;
bool is_devMem = true;
static int mallocHostFlag=0;
static int mallocFlag=0;

void* deleteReslutDir(){
    system("mkdir -p result_Files");
    system("rm -rf result_Files/*");
}
aclError testaclInit(const char *configPath){
    error = aclInit(configPath);
    if (error != ACL_ERROR_NONE) {
        ACL_LOG("aclInit failed, error[%d]",error);
        return error;
    }
    ACL_LOG(" aclInit success,error[%d]\n",error);
    return error;
}
aclError aclDeviceContexInit(uint32_t devNum, uint32_t device_id, std::vector<aclrtContext>& contex_vec){
    aclError ret = ACL_ERROR_NONE;
    
    for (int devIndex = device_id; devIndex < devNum+device_id; devIndex++)
    {
        ret = aclrtSetDevice(devIndex);
        if (ret != ACL_ERROR_NONE)
        {
            printf("[ERROR]aclrtSetDevice failed, ret %d\n", ret);
            return ret;
        }
        
        aclrtContext context;
        ret = aclrtCreateContext(&context, devIndex);
        if (ret != ACL_ERROR_NONE)
        {
            printf("[ERROR]aclrtCreateContext failed, ret %d\n", ret);
            return ret;
        }

        contex_vec.push_back(context);
    }

    return ret;
    
}
void aclModelUnloadAndDescDestroy(uint32_t modelId, aclmdlDesc* modelDesc){
    aclError ret;
    
    ret = aclmdlUnload(modelId);
    if (ret != ACL_ERROR_NONE) 
    {
        printf("aclmdlUnload  failed, ret[%d]\n", ret);
    }
    printf("unload model success\n");

    ret = aclmdlDestroyDesc(modelDesc);
    if (ret != ACL_ERROR_NONE) 
    {
        printf("aclmdlDestroyDesc  failed, ret[%d]\n", ret);
    }
}

void aclDeviceContexDestroy(uint32_t devNum, uint32_t device_id, std::vector<aclrtContext>& contex_vec){
    aclError ret = ACL_ERROR_NONE;
    
    for (int devIndex = device_id; devIndex < devNum+device_id; devIndex++)
    {
        aclrtResetDevice(devIndex);
    }

    for (auto iter = contex_vec.begin(); iter != contex_vec.end(); iter++)
    {
        aclrtDestroyContext(*iter);
    }
    aclFinalize();
    return ;
    
}
void aclReleaseAllModelResource(uint32_t device_id, uint32_t modelId,  aclmdlDesc* modelDesc, std::vector<aclrtContext>& contex_vec){
    if (modelDesc && modelId)
    {
        aclModelUnloadAndDescDestroy(modelId, modelDesc);
    }
    aclDeviceContexDestroy(deviceNum, device_id,contex_vec);
}

aclError testACL_ModelInputCreate(){
    ACL_LOG("ACL_ModelInput Create start");
    modelInput = aclmdlCreateDataset();
    if(modelInput == NULL) {
        ACL_LOG("ACL_ModelInput Create failed");
        ASSERT_EQ(ACL_ERROR_NONE, 1);
    }
    ACL_LOG("ACL_ModelInput Create finish,addr[%p]\n",modelInput);
    return error;
}
aclError testACL_ModelInputDataBuffInit(){
    ACL_LOG("ACL_ModelInputDataBuff Init start");

    size_t inputTensorDescNum = aclmdlGetNumInputs( modelDesc);
    if(inputTensorDescNum==0){
        error+=1;
        ACL_LOG("aclmdlGetNumInputs failed, inputTensorDescNum=%u,not expect[0]",inputTensorDescNum);
        ASSERT_NE(0, inputTensorDescNum);
    }

    for(size_t i = 0; i < inputTensorDescNum; ++i) {
        size_t buffer_size = aclmdlGetInputSizeByIndex(modelDesc, i);
        void* inputBuffer = NULL;
        if(is_devMem){
            error = aclrtMalloc(&inputBuffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
            mallocFlag++;
        }
        if (error != ACL_ERROR_NONE) {
            ACL_LOG("aclrtMalloc failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("aclrtMalloc,inputBuffer:%p,size:%lu", inputBuffer, buffer_size);

        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, buffer_size);
        if(inputData == NULL) {
            ACL_LOG("aclCreateDataBuffer failed");
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("aclDataBuffer:%p", inputData);

        error = aclmdlAddDatasetBuffer(modelInput, inputData);
        if(error != ACL_ERROR_NONE) {
            ACL_LOG("ACL_ModelInputDataAdd failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
    }
    ACL_LOG("ACL_ModelInputDataBuff init finish\n");
    return error;
}
aclError testACL_ModelOutputCreate(){
    ACL_LOG("ACL_ModelOutputCreate start");

    modelOutput = aclmdlCreateDataset();
    if(modelOutput == NULL) {
        ACL_LOG("ACL_ModelOutputCreate failed");
        ASSERT_EQ(ACL_ERROR_NONE, 1);
    }
    ACL_LOG("ACL_ModelOutputCreate finishaddr[%p]\n",modelOutput);
    return error;
}
aclError testACL_ModelOutputDataBuffInit(){
    ACL_LOG("ACL_ModelOutputDataBuffInit start");
    size_t outputTensorDescNum = aclmdlGetNumOutputs( modelDesc);
    if(outputTensorDescNum==0){
        ACL_LOG("aclmdlGetNumOutputs failed, outputTensorDescNum=%u,not expect[0]",outputTensorDescNum);
        ASSERT_NE(0, outputTensorDescNum);
    }

    for(size_t i = 0; i < outputTensorDescNum; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc, i);
        void* outputBuffer = NULL;
        if(is_devMem){
            error = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
            mallocFlag++;
        }else{
            error = aclrtMallocHost(&outputBuffer, buffer_size);
            mallocHostFlag++;
        }
        if (error != ACL_ERROR_NONE) {
            ACL_LOG("aclrtMalloc failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("aclrtMalloc:output:%p,size:%lu", outputBuffer, buffer_size);

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if(outputData == NULL) {
            ACL_LOG("aclCreateDataBuffer failed");
            ASSERT_EQ(ACL_ERROR_NONE, 1);
        }
        error = aclmdlAddDatasetBuffer(modelOutput, outputData);
        if(error != ACL_ERROR_NONE) {
            ACL_LOG("ACL_ModelOutputDataAdd failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
    }
    ACL_LOG("ACL_ModelOutputDataBuffInit finish\n");
    return error;
}
aclError testPushData2ModelMultiBatchSingleInputDataBuff(aclmdlDataset * model_input, std::string inputPath,std::vector<std::string>& file_vec, size_t batchSize ){
    ACL_LOG("testPushData2ModelMultiBatchSingleInputDataBuff start");
    //??tensor????
    //inputdata0    
    void* p_batchDst =NULL;
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(model_input, 0);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    size_t modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    uint32_t singleBuffSize = modelInputSize / batchSize;
    uint32_t pos = 0;
    aclError ret = 0;

    for(int i=0; i < file_vec.size(); i++)
    {
        std::string fileName = inputPath + "/" + file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        ACL_LOG("open file name : %s ",fileName.c_str());
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        } 



        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d]\n", ret);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);
        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d errno : %d ", ret,__LINE__,errno);

            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;

        aclrtFreeHost(p_imgBuf);
    }

    ACL_LOG("testPushData2ModelMultiBatchSingleInputDataBuff end");
    return 0;
}

aclError testaclmdlExecute(uint32_t model_id, aclmdlDataset *model_input, aclmdlDataset *model_output ){
    error = aclmdlExecute(model_id, model_input, model_output);
    if(error != ACL_ERROR_NONE) {
        ACL_LOG("aclmdlExecute failed, ret[%d]", error);
    }
    return error;
}
/*
*ACL_ModelOutput * model_output 
*uint32_t batchSize ?batch???????tensor????batch???
*vector<std::string>& v_inferFile  ?batch???????
*/
aclError testPullDataFromModelMultiBatchOutputDataBuff(aclmdlDataset * model_output, std::vector<std::string>& v_inferFile,size_t batchSize ){
    ACL_LOG("testPullDataFromModelMultiBatchOutputDataBuff start");

    std::string retFolder = "./result_Files";
    DIR* op = opendir(retFolder.c_str());
    if (NULL == op)
    {
        mkdir(retFolder.c_str(), 00775);
    }
    else
    {
        closedir(op);
    }
    //?output???????
    size_t outDatasetNum=aclmdlGetDatasetNumBuffers(model_output);
    if(outDatasetNum==0){
        ACL_LOG("aclmdlGetDatasetNumBuffers from model_output failed, outDatasetNum=%u,not expect[0]",outDatasetNum);
        ASSERT_NE(0, outDatasetNum);
    }
    for (size_t i = 0; i < outDatasetNum; ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(model_output, i);
        if(dataBuffer==NULL){
            ACL_LOG("aclmdlGetDatasetBuffer from model_output failed, dataBuffer=%p,not expect[NULL]",dataBuffer);
            ASSERT_NE(NULL, dataBuffer);
        }

        void* data = aclGetDataBufferAddr(dataBuffer);
        if(data==NULL){
            ACL_LOG("aclGetDataBufferAddr from dataBuffer failed, data_addr=%p,not expect[NULL]",data);
            ASSERT_NE(NULL, data);
        }
        size_t bufferSize = aclGetDataBufferSize(dataBuffer);
        ACL_LOG("output[%zu] DataBuffer, buffer data:[%p], buffer size:[%zu]",i,data,bufferSize);

        //??host??
        void* hostPtr=NULL;
        error = aclrtMallocHost(&hostPtr, bufferSize);
        if(error != ACL_ERROR_NONE) {
            ACL_LOG("aclrtMallocHost failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("aclrtMallocHost success, error=[%d],addr=[%p],size=[%zu]", error, hostPtr, bufferSize);

        //????
        error = aclrtMemcpy(hostPtr, bufferSize,data,bufferSize , ACL_MEMCPY_DEVICE_TO_HOST);
        if(error != ACL_ERROR_NONE) {
            ACL_LOG("aclrtMemcpy D2H failed, ret[%d], errno[%d]", error,errno);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("memcopy output data to host buffer,D2H,error[%d]",error);
        //data ???
        uint32_t len=(uint32_t)bufferSize/batchSize;
        ACL_LOG("write file,filesize:%d, bufferSize: %d, file num:%d\n", len, bufferSize, v_inferFile.size());
        for (size_t j = 0; j < v_inferFile.size(); j++)
        {
            std::string framename = v_inferFile[j];
            std::size_t dex = (framename).find_last_of(".");
            std::string inputFileName = (framename).erase(dex);

            FILE* outputFile = fopen((retFolder + "/" + "davinci_" + inputFileName + "_"  + "output" + std::to_string(i) + ".bin").c_str(), "wb");
            if (NULL == outputFile)
            {
                //out.close();
                aclrtFreeHost(hostPtr);
                return 1;
            }

            fwrite((uint8_t *)hostPtr + (j * len), len, sizeof(char), outputFile);
            fclose(outputFile);
    }

        //?????host??
        error = aclrtFreeHost(hostPtr);
        if(error != ACL_ERROR_NONE) {
            ACL_LOG("aclrtFreeHost failed, ret[%d]", error);
            ASSERT_EQ(ACL_ERROR_NONE, error);
        }
        ACL_LOG("aclrtFreeHost success, error=[%d]", error);
    }
    ACL_LOG("testPullDataFromModelMultiBatchOutputDataBuff finish\n");
    return error;
}


aclError testPushData2ModelMultiBatch8InputDataBuff(aclmdlDataset * model_input, std::string inputPath,std::vector<std::string>& input1_file_vec,std::vector<std::string>& input2_file_vec,std::vector<std::string>& input3_file_vec,std::vector<std::string>& input4_file_vec,std::vector<std::string>& input5_file_vec,std::vector<std::string>& input6_file_vec,std::vector<std::string>& input7_file_vec,std::vector<std::string>& input8_file_vec, size_t batchSize ){
    ACL_LOG("testPushData2ModelMultiBatch8InputDataBuff start");
    //获取tensor内存大小

    void* p_batchDst =NULL;
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(model_input, 0);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    size_t modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    uint32_t singleBuffSize = modelInputSize / batchSize;
    uint32_t pos = 0;
    aclError ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input1_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_1/" + input1_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input1_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }

    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 1);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input2_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_2/" + input2_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input2_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }

    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 2);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input3_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_3/" + input3_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input3_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }



    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 3);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input4_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_4/" + input4_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input4_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }


    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 4);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input5_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_5/" + input5_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input5_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }


    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 5);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input6_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_6/" + input6_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input6_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }


    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 6);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input7_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_7/" + input7_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input7_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }

    p_batchDst =NULL;
    dataBuffer = aclmdlGetDatasetBuffer(model_input, 7);
    p_batchDst = aclGetDataBufferAddr(dataBuffer);
    modelInputSize =(size_t) aclGetDataBufferSize(dataBuffer);
    singleBuffSize = modelInputSize / batchSize;
    pos = 0;
    ret = 0;
    ACL_LOG("input0 size: %d ",singleBuffSize);

    for(int i=0; i < input8_file_vec.size(); i++)
    {
        std::string fileName = inputPath + "input_8/" + input8_file_vec[i];
        FILE * pFile = fopen(fileName.c_str(),"r");
        if (NULL == pFile)
        {
            ACL_LOG("open file %s failed\n", input8_file_vec[i].c_str());
            continue;
        }
        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);
        rewind(pFile);

        if (fileSize != singleBuffSize)
        {
            printf("[ERROR] index %d file %s size[%d] not equal to model input size[%d], skip to next file\n", i, fileName.c_str(), fileSize, singleBuffSize);
            fclose(pFile);
            continue;
        }


        void* p_imgBuf = NULL;
        ret = aclrtMallocHost(&p_imgBuf, fileSize);

        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("p_imgBuf aclrtMallocHost failed[%d] errno[%d]\n", ret,errno);
            fclose(pFile);
            continue;
        }
        fread((uint8_t *)p_imgBuf, sizeof(char), fileSize, pFile);
        fclose (pFile);

        ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, p_imgBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG("aclrtMemcpy failed, ret[%d] line : %d, errno :%d", ret,__LINE__,errno);
            aclrtFreeHost(p_imgBuf);
            continue;
        }
        pos += fileSize;
        ACL_LOG("fileName: %s fileSize: %d", fileName.c_str(),fileSize);
        aclrtFreeHost(p_imgBuf);
    }




    ACL_LOG("testPushData2ModelMultiBatch8InputDataBuff end");
    return 0;
}


aclError testaclmdlLoadFromMem(char* modelData, uint32_t modelSize, uint32_t* model_id){
    ACL_LOG("aclmdlLoadFromMem start");
    error = aclmdlLoadFromMem(modelData, modelSize, model_id);
    if (error != ACL_ERROR_NONE) {
        ACL_LOG("aclmdlLoadFromMem failed, ret[%d]", error);
    }
    ACL_LOG("aclmdlLoadFromMem finish, modelId[%u], ret[%d]\n",*model_id, error);
    return error;
}
aclError testaclmdlDescInit(){
    modelDesc = aclmdlCreateDesc();
    if (modelDesc== NULL) {
        error+=1;
        ACL_LOG("aclmdlCreateDesc failed,error[%d]",error);

    }
    else{
        ACL_LOG(" aclmdlCreateDesc success");
        error += aclmdlGetDesc(modelDesc, modelId);
        if (error != ACL_ERROR_NONE) {
            ACL_LOG("aclmdlGetDesc faild, error[%d]",error);
        }
        ACL_LOG(" aclmdlGetDesc success, error[%d]",error);
    }
    //??acl?????model tensor??
    printf("************************************************************\n");
    size_t inputDescSize = aclmdlGetNumInputs(modelDesc);
    size_t outputDescSize = aclmdlGetNumOutputs(modelDesc);
    printf("modelDesc tensor size info:\n");
    for (size_t i = 0; i < inputDescSize; ++i) {
        size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc, i);
        const char* inputname = aclmdlGetInputNameByIndex(modelDesc, i);
        aclFormat inputformat = aclmdlGetInputFormat(modelDesc, i);
        aclDataType inputdataType = aclmdlGetInputDataType(modelDesc, i);
        printf("index[%zu], name:[%s], inputSize[%zu], format[%d], dataType[%d]\n", i, inputname, inputSize, (int)inputformat, (int)inputdataType);

        aclmdlIODims ioDims;
        aclmdlGetInputDims(modelDesc, i, &ioDims);
        ASSERT_EQ(ACL_ERROR_NONE, error);
    for(size_t j = 0; j < ioDims.dimCount; ++j){
          printf("ioDims:[%ld]\n", ioDims.dims[j]);
        }
    }
    for (size_t i = 0; i < outputDescSize; ++i) {
        printf("output[%zu-%zu]:    tensor size:[%zu]\n", outputDescSize, i, aclmdlGetOutputSizeByIndex(modelDesc, i));
        size_t OutputSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        const char* outputname = aclmdlGetOutputNameByIndex(modelDesc, i);
        aclFormat outputformat = aclmdlGetOutputFormat(modelDesc, i);
        aclDataType outputdataType = aclmdlGetOutputDataType(modelDesc, i);
        printf("index[%zu], name:[%s], outputSize[%zu], format[%d], dataType[%d]\n", i, outputname, OutputSize, (int)outputformat, (int)outputdataType);

        aclmdlIODims ioDims;
        aclmdlGetOutputDims(modelDesc, i, &ioDims);
        ASSERT_EQ(ACL_ERROR_NONE, error);
    for(size_t j = 0; j < ioDims.dimCount; ++j){
          printf("[%ld]\n", ioDims.dims[j]);
        }

    }
    printf("************************************************************\n\n");
    return error;
}
char* ReadBinFile(std::string fileName, uint32_t& fileSize){
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        printf("open file[%s] failed\n", fileName.c_str());
        return NULL;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        printf("binfile is empty, filename: %s", fileName.c_str());
        binFile.close();
        return NULL;
    }

    binFile.seekg(0, binFile.beg);

    char* binFileBufferData = new(std::nothrow) char[binFileBufferLen];
    if (binFileBufferData == NULL) {
        printf("malloc binFileBufferData failed\n");
        binFile.close();
        return NULL;
    }
    binFile.read(binFileBufferData, binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}
int scanFiles(std::vector<std::string> &fileList, std::string inputDirectory){
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char* str = inputDirectory.c_str();

    p_dir = opendir(str);
    if( p_dir == NULL )
    {
                printf("[ERROR] Open directory[%s] failed. \n",str);
        return -1;
    }

    struct dirent *p_dirent;

    while ( p_dirent = readdir(p_dir))
    {
        std::string tmpFileName = p_dirent->d_name;

        if( tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else if (p_dirent->d_type == 8)
        {
            //file
            fileList.push_back(tmpFileName);
        }
        else if (p_dirent->d_type == 10)
        {
            //link file
            continue;
        }
        else if (p_dirent->d_type == 4)
        {
            //dir
            continue;
        }
        else
        {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);

    if( fileList.size() == 0 )
    {
                printf("[ERROR] No file in the directory[%s]",str);
    }
    return fileList.size();
}


int main(int argc, char** argv)
{

    const char* input_om;
    const char* input_data;
    int input_batchsize;
    if (4 == argc) {
        input_om = argv[1];
        input_data = argv[2];
        input_batchsize = atoi(argv[3]);
        printf("om file: [%s]\n", input_om);
        printf("input data: [%s]\n", input_data);
        printf("batchsize: [%d]\n", input_batchsize);
     }
     else
     {
         printf("you should input  three argument, which means the om file/input data/batchsize!\n");
         printf("for example(two input data):  ./main resnet50.om  input_data  batchsize \n");
         return -1;
     }
    uint32_t device_id = 0;
    char *id;
    if (id=getenv("DEVICE_ID"))    //从环境变量中获取DEVICE_ID
    {
        device_id = atoi(id);
    }
    else
    {
        device_id = 0;
        ACL_LOG("Can't find DEVICE_ID in env, set device_id=[%d]", device_id);
    }
    ACL_LOG("aclSetDevice, device_id=[%d]", device_id);


    aclError ret;
    //????????
    deleteReslutDir();
    timeval start, end;
    // ???,???????configPath??
    const char *configPath = "";
    error = testaclInit(configPath);
    ASSERT_EQ(ACL_ERROR_NONE, error);
    std::vector<aclrtContext> contex_vec;
    ret = aclDeviceContexInit(deviceNum, device_id, contex_vec);
    if (ret != ACL_ERROR_NONE)
    {
        printf("aclDeviceContexInit failed, ret[%d]\n", ret);
        return 1;
    }
    printf("[step 1] initial success\n");

    int pic_num = 0;

    //??model
    std::string om_str = input_om;
    char model_name[128];
    strcpy(model_name,om_str.c_str());
    ACL_LOG("om file is:%s",model_name);

    // ????
    const char* modelFile=(const char*)model_name;
    ACL_LOG("start to load om to mem,om File is:[%s]",modelFile);
    uint32_t modelSize = 0;
    char* modelData = ReadBinFile(modelFile, modelSize);
    if (modelData == NULL) {
        ACL_LOG("ReadBinFile failed");
        ASSERT_EQ(ACL_ERROR_NONE, 1);
    }
    testaclmdlLoadFromMem(modelData, modelSize, &modelId);
    //ASSERT_EQ(ACL_ERROR_NONE, error);
    // ????????
    testaclmdlDescInit();

    //???????
    testACL_ModelInputCreate();
    testACL_ModelInputDataBuffInit();
    testACL_ModelOutputCreate();
    testACL_ModelOutputDataBuffInit();

    std::string inputFile= input_data;
    std::string input1_file_name;
    std::string input2_file_name;
    std::string input3_file_name;
    std::string input4_file_name;
    std::string input5_file_name;
    std::string input6_file_name;
    std::string input7_file_name;
    std::string input8_file_name;
    std::vector<std::string> fileName_vec;
    std::vector<std::string> input1_inferFile_vec;
    std::vector<std::string> input2_inferFile_vec;
    std::vector<std::string> input3_inferFile_vec;
    std::vector<std::string> input4_inferFile_vec;
    std::vector<std::string> input5_inferFile_vec;
    std::vector<std::string> input6_inferFile_vec;
    std::vector<std::string> input7_inferFile_vec;
    std::vector<std::string> input8_inferFile_vec;
    int fileNums=0;
    fileNums = scanFiles(fileName_vec, inputFile + "input_1");
    float total = 0.0;
    int i=0;
    int batchSize = input_batchsize;
    for(i=0;i<fileName_vec.size();++i)
    {
        input1_file_name = "input_1_" + std::to_string(i) + ".bin";
        input2_file_name = "input_2_" + std::to_string(i) + ".bin";
        input3_file_name = "input_3_" + std::to_string(i) + ".bin";
        input4_file_name = "input_4_" + std::to_string(i) + ".bin";
        input5_file_name = "input_5_" + std::to_string(i) + ".bin";
        input6_file_name = "input_6_" + std::to_string(i) + ".bin";
        input7_file_name = "input_7_" + std::to_string(i) + ".bin";
        input8_file_name = "input_8_" + std::to_string(i) + ".bin";
        input1_inferFile_vec.push_back(input1_file_name);
        input2_inferFile_vec.push_back(input2_file_name);
        input3_inferFile_vec.push_back(input3_file_name);
        input4_inferFile_vec.push_back(input4_file_name);
        input5_inferFile_vec.push_back(input5_file_name);
        input6_inferFile_vec.push_back(input6_file_name);
        input7_inferFile_vec.push_back(input7_file_name);
        input8_inferFile_vec.push_back(input8_file_name);
        if( (i+1) % batchSize == 0)
        {
            //???? 1:1input 2:2input
            testPushData2ModelMultiBatch8InputDataBuff(modelInput,inputFile,input1_inferFile_vec,input2_inferFile_vec,input3_inferFile_vec,input4_inferFile_vec,input5_inferFile_vec,input6_inferFile_vec,input7_inferFile_vec,input8_inferFile_vec, batchSize);

            //????
            gettimeofday(&start, NULL);
            testaclmdlExecute(modelId, modelInput, modelOutput);
            gettimeofday(&end, NULL);
            //ASSERT_EQ(ACL_ERROR_NONE, error);

            // ????
            testPullDataFromModelMultiBatchOutputDataBuff(modelOutput, input1_inferFile_vec, batchSize);
            total += 1000*(end.tv_sec - start.tv_sec) + 1.0*(end.tv_usec - start.tv_usec)/1000;
            pic_num += input1_inferFile_vec.size();
            input1_inferFile_vec.clear();
            input2_inferFile_vec.clear();
            input3_inferFile_vec.clear();
            input4_inferFile_vec.clear();
            input5_inferFile_vec.clear();
            input6_inferFile_vec.clear();
            input7_inferFile_vec.clear();
            input8_inferFile_vec.clear();

        }
    }
    if( i % batchSize != 0)
    {
        //????
        testPushData2ModelMultiBatch8InputDataBuff(modelInput,inputFile,input1_inferFile_vec,input2_inferFile_vec,input3_inferFile_vec,input4_inferFile_vec,input5_inferFile_vec,input6_inferFile_vec,input7_inferFile_vec,input8_inferFile_vec, batchSize);
        //????
        gettimeofday(&start, NULL);
        testaclmdlExecute(modelId, modelInput, modelOutput);
        gettimeofday(&end, NULL);
        //ASSERT_EQ(ACL_ERROR_NONE, error);
        //????
        testPullDataFromModelMultiBatchOutputDataBuff(modelOutput, input1_inferFile_vec, batchSize);
        total += 1000*(end.tv_sec - start.tv_sec) + 1.0*(end.tv_usec - start.tv_usec)/1000;
        pic_num += batchSize;
        input1_inferFile_vec.clear();
        input2_inferFile_vec.clear();
        input3_inferFile_vec.clear();
        input4_inferFile_vec.clear();
        input5_inferFile_vec.clear();
        input6_inferFile_vec.clear();
        input7_inferFile_vec.clear();
        input8_inferFile_vec.clear();
    }
    //?????modeldata
    delete[] modelData;
    ACL_LOG("buffer free finish");
    aclReleaseAllModelResource(device_id, modelId, modelDesc, contex_vec);

    //????
    ACL_LOG("Inference time cost:%.2lf ms",total)
    std::cout<<"total cost: "<<total<<", pic number: "<<pic_num<<std::endl;
    ACL_LOG("Average inference time cost:%.2lf ms",(1.0*total)/pic_num);
    //ASSERT_LT((1.0*total)/pic_num,expect_time);
}
