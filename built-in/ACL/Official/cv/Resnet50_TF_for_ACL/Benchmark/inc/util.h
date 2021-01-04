#ifndef BENCHMARK_UTIL_H
#define BENCHMARK_UTIL_H
#include <string>
#include <stdio.h>
#include <sys/time.h>
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/acl_rt.h"
#include "acl/ops/acl_dvpp.h"
#include <iostream>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <unordered_map>
#include <utility>

// self defined problem code.
const int ACL_ERROR_PATH_INVALID = 101;
const int ACL_ERROR_CREATE_DATASET_FAILED = 102;
const int ACL_ERROR_PARSE_PARAM_FAILED = 103;
const int ACL_ERROR_DVPP_ERROR = 104;
const int ACL_ERROR_OTHERS = 255;
#define MODEL_INPUT_NUM_MAX (4)
#define MODEL_INPUT_OUTPUT_NUM_MAX (16)

#define LOG(fmt, args...)       \
	do {                        \
		printf(fmt, ##args);    \
	} while(0)


#define START_PROC                      \
	struct timeval start, end;          \
	long long time_use;                 \
	do {                                \
		gettimeofday(&start, NULL);     \
	} while (0);


#define END_PROC                                                                    \
	do {                                                                            \
		gettimeofday(&end, NULL);                                                   \
		time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);   \
		LOG("time use: %lld us\n", time_use); 						        \
	} while (0);


#define CHECK_ACL_RET(msg, ret) 																\
	if (ret != ACL_ERROR_NONE) {																\
		std::cout << msg << ", ret "<< ret << std::endl;	\
		return ret;																				\
	}


#define CHECK_WITH_RET(condition, ret, msg)														\
	if(!(condition)) {																			\
		std::cout << msg << ", ret "<< ret << std::endl;	\
		return ret;																				\
	}


#define CHECK_RET(ret)			\
	if(ret != ACL_ERROR_NONE) {	\
		return ret;				\
	}

bool FolderExists(std::string foldname);

bool FileExists(std::string filename);

char* ReadBinFile(std::string fileName, uint32_t& fileSize);

aclError GetFiles(std::string path, std::vector<std::string>& files);

aclError FreeDevMemory(aclmdlDataset* dataset);

aclError DestroyDatasetResurce(aclmdlDataset* dataset, uint32_t flag);

void* ReadFile(std::string fileLocation, uint64_t &fileSize);

struct DvppConfig {
	uint32_t resizedWidth;
	uint32_t resizedHeight;
	std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> imgSizes;
};

struct ModelInfo
{
    aclFormat Format;
    const char* Name;
    size_t size;
    size_t dimCount;
    int64_t dims[ACL_MAX_DIM_CNT];
    aclDataType Type;
};

struct Config {
    std::string om;
    std::string dataDir;
    std::string outDir;
    DvppConfig dvppConfig;
    bool useDvpp;
    size_t batchSize;
    ModelInfo inputInfo[MODEL_INPUT_OUTPUT_NUM_MAX];
    ModelInfo outputInfo[MODEL_INPUT_OUTPUT_NUM_MAX];
    size_t inputNum;
    size_t outputNum;
    aclmdlDesc* modelDesc;
    uint32_t  modelId;
    aclrtContext context;
    char* modelData_ptr;
    void* devMem_ptr;
    void* weightMem_ptr;
    std::string imgType;
    std::string modelType;
    uint32_t deviceId;
    uint32_t loopNum;
    std::string framework;
    int64_t curOutputSize[MODEL_INPUT_OUTPUT_NUM_MAX];
    Config()
    {
        om = "";
        dataDir = "";
        batchSize = 0;
        useDvpp = 0;
        inputNum = 0;
        outputNum = 0;
        modelDesc = nullptr;
        modelId = 0;
        context = nullptr;
        imgType = "";
        modelType = "";
        deviceId = 0;
        loopNum = 1;
        framework = "caffe";
        outDir = "../../results";
        modelData_ptr = nullptr;
        devMem_ptr = nullptr;
        weightMem_ptr = nullptr;
    }
};

struct Resnet50Result {
	int top1;
	int top5;
	int total;
	std::unordered_map<std::string, int> cmp;
	Resnet50Result(): top1(0), top5(0), total(0) {};
};

struct DataFrame {
    std::vector<std::string> fileNames;
    aclmdlDataset* dataset;
};

#endif
