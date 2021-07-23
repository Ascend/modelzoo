#ifndef VISION_PREPROCESS_H
#define VISION_PREPROCESS_H

#include <vector>
#include <string>
#include <thread>
#include <sys/time.h>
#include "common/data_struct.h"
#include <cstdint>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "dvpp_process.h"




class VisionPreProcess{
public:
	VisionPreProcess();
	~VisionPreProcess();
	int Init(int input_width, int input_height);
    void DeInit();
    void Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, BlockingQueue<std::shared_ptr<ModelInputData>>*
	outputQueuePtr);
    std::shared_ptr<PerfInfo> GetPerfInfo();

private:
	 
    
    //int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    DvppProcess *processDvpp;
    int modelInputWidth;
    int modelInputHeight;

    bool isStop_ = false;
    std::thread processThr_;
    void ProcessThread();

    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr_ = nullptr;
    BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr_ = nullptr;

    void DestroyResource();
    void*GetDeviceBufferOfPicture(const char* inputHostBuff,const uint32_t inputHostBuffSize,uint32_t &devPicBufferSize);

    //for perf info
    std::shared_ptr<PerfInfo> perfInfo_;
    uint64_t GetCurentTimeStamp();
    struct timeval currentTimeval;
    uint64_t initialTimeStamp;
};

#endif

