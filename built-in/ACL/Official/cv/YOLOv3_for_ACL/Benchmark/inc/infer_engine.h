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
acldvppRoiConfig* InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight,uint32_t modelInputWidth,
                                          uint32_t modelInputHeight);
/*
 * @brief : 宽高较长的边缩放至RESIZE_MAX(416)，较短的边做等比例缩放。
 * @param [in] uint32_t width : 输入文件路径.
 * @param [in] uint32_t height : 输出buffer指针.
 * @param [in] uint32_t &newInputWidth : 等比例缩放后的宽
 * @param [in] uint32_t &newInputHeight : 等比例缩放后的高
 */
void SmallSizeAtLeast(uint32_t width, uint32_t height, uint32_t& newInputWidth, uint32_t& newInputHeigh);

#endif
