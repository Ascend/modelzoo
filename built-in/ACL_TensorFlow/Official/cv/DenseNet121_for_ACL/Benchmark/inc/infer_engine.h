#ifndef BENCHMARK_INFER_ENGINE_H
#define BENCHMARK_INFER_ENGINE_H
#include "util.h"
#include "acl/acl_base.h"
#include "post_process.h"
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include "acl/acl_mdl.h"
#include <thread>
#include <memory>

aclError InitContext(const char* configPath = "");
aclError UnInitContext();
aclError LoadModel();
aclError InitInput(std::vector<std::string> files);
aclError Inference();
aclError PostProcess();
aclError DvppSetup();
aclError DvppInitInput(std::vector<std::string> files);
aclError UnloadModel();
void getImgResizeShape();
aclError GetImageSize(std::string file);

aclError DVPP_Densenet121(std::string fileLocation, char *&ptr);
void GetImageHW(void* buff, uint32_t fileSize, std::string fileLocation, uint32_t &W, uint32_t &H);



#endif
