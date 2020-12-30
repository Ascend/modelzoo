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
acldvppRoiConfig* InitCropRoiConfig(uint32_t width, uint32_t height);
acldvppRoiConfig* InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight,uint32_t modelInputWidth, uint32_t modelInputHeight);
void SmallSizeAtLeast(uint32_t width, uint32_t height, uint32_t& newInputWidth, uint32_t& newInputHeigh);
#endif
