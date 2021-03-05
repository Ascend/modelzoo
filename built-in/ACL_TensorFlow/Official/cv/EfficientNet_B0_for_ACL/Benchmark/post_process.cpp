#include "post_process.h"
#include "util.h"
#include <vector>
#include <cstring>
#include <memory>
#include <fstream>
#include "stdio.h"
#include <sys/time.h>
#include <unistd.h>
#include <time.h> 
#include <dirent.h> 
#include <stdarg.h>
#include <iostream>

#include <unistd.h>
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
#include <unordered_map>


extern int processedCnt;

extern Config cfg;
extern DataFrame outputDataframe;
extern aclError ret;
int topNum = 5;

extern int processedCnt;

aclError SaveBinPostprocess()
{
    aclError retVal;

    DataFrame dataframe = outputDataframe;
    std::vector<std::string>& inferFile_vec = outputDataframe.fileNames;
    aclmdlDataset* output = dataframe.dataset;

    std::string resultFolder = cfg.outDir + "/" + cfg.modelType;
    DIR* op = opendir(resultFolder.c_str());
    if (NULL == op){
        mkdir(resultFolder.c_str(), 00775);
    }else{
        closedir(op);
    }

    for (size_t i = 0; i < cfg.outputNum; ++i)
    {        
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len;
        len = cfg.outputInfo[i].size;
        
        void* outHostData = NULL;
        ret = aclrtMallocHost(&outHostData, len);
        CHECK_ACL_RET("malloc host failed", ret);
  
        ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            std::cout  << "Copy device to host failed" << std::endl;
            aclrtFreeHost(outHostData);
            return 1;
        }

        uint32_t eachSize = len / cfg.batchSize;
        for (size_t j = 0; j < inferFile_vec.size(); j++)
        {
            FILE* outputFile;
            std::string framename = inferFile_vec[j];
            std::size_t dex = (framename).find_first_of(".");
            std::string inputFileName = (framename).erase(dex);
            
            if (cfg.modelType.compare(0, 6, "resnet") == 0){
                outputFile = fopen((resultFolder + "/" + "davinci_" + inputFileName + "_"  + "output" + ".bin").c_str(), "wb");
            }else{
                outputFile = fopen((resultFolder + "/" + "davinci_" + inputFileName + "_"  + "output" + std::to_string(i) + ".bin").c_str(), "wb");
            }
            
            if (NULL == outputFile){
                aclrtFreeHost(outHostData);
                return 1;
            }

            fwrite((uint8_t *)outHostData + (j * eachSize), eachSize, sizeof(char), outputFile);
            fclose(outputFile);
        }
        
        ret = aclrtFreeHost(outHostData);
        CHECK_ACL_RET("free output host fialed", ret);
    }
    
    (void)DestroyDatasetResurce(outputDataframe.dataset, 0);
    
    std::cout << "save batch " << processedCnt << " done " << std::endl;
    return ACL_ERROR_NONE;
}
