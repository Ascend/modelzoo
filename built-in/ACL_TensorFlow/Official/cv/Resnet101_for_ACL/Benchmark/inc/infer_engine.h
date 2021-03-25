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

/**
 * @brief init context
 * @param [in] configPath: path of config file
 * @return result
 **/
aclError InitContext(const char* configPath = "");

/**
 * @brief uninit context
 * @return result
 **/
aclError UnInitContext();

/**
 * @brief load model
 * @return result
 **/
aclError LoadModel();

/**
 * @brief init input for no dvpp case
 * @param [in] files: files name of input pic
 * @return result
 **/
aclError InitInput(std::vector<std::string> files);

/**
 * @brief model execute
 * @return result
 **/
aclError Inference();
aclError PostProcess();

/**
 * @brief create dvpp channel desc
 * @return result
 **/
aclError DvppSetup();

/**
 * @brief dvpp config
 * @param [in] files: files name of input pic
 * @return result
 **/
aclError DvppInitInput(std::vector<std::string> files);
aclError UnloadModel();
void getImgResizeShape();
acldvppRoiConfig* InitCropRoiConfig(uint32_t width, uint32_t height);
acldvppRoiConfig* InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight,uint32_t modelInputWidth, uint32_t modelInputHeight);
void SmallSizeAtLeast(uint32_t width, uint32_t height, uint32_t& newInputWidth, uint32_t& newInputHeigh);



#endif
