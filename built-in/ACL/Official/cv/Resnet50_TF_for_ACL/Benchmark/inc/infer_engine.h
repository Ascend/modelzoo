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

/*
 * @brief : 初始化中心抠图配置信息。
 * @param [in] uint32_t newInputWidth : 输入图像的宽（等比例缩放后的宽度）
 * @param [in] uint32_t newInputHeight : 输入图像的高（等比例缩放后的高）
 * @param [in] uint32_t modelInputWidth : 中心抠图后输入给模型的宽
 * @param [in] uint32_t modelInputHeight : 中心抠图后输入给模型的高
 * @return : acldvppRoiConfig：中心抠图配置信息
 */
acldvppRoiConfig* InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight,uint32_t modelInputWidth, uint32_t modelInputHeight);

/*
 * @brief : 宽高较短的边缩放至RESIZE_MIN(256)，较长的边做等比例缩放。
 * @param [in] uint32_t width : 输入图片宽
 * @param [in] uint32_t height : 输入图片高
 * @param [in] uint32_t &newInputWidth : 等比例缩放后的宽
 * @param [in] uint32_t &newInputHeight : 等比例缩放后的高
 */
void SmallSizeAtLeast(uint32_t width, uint32_t height, uint32_t& newInputWidth, uint32_t& newInputHeigh);
#endif
