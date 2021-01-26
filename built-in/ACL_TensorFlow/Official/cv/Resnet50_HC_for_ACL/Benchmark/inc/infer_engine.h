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

/**
 * @brief unload model
 * @return result
 **/
aclError UnloadModel();

/**
 * @brief get the images resize shape, that's model input shape
 **/
void getImgResizeShape();

/**
 * @brief create dvpp picture desc
 * @param [in] dataDev: ptr of image memory address
 * @param [in] format: enum of acldvppPixelFormat
 * @param [in] width: width before alignment
 * @param [in] height: height before alignment
 * @param [in] widthStride: width after alignment
 * @param [in] heightStride: height after alignment
 * @param [in] size: size of dataDev
 * @return acldvppPicDesc
 **/
acldvppPicDesc *createDvppPicDesc(void *dataDev, acldvppPixelFormat format, uint32_t width, uint32_t height, uint32_t widthStride, uint32_t heightStride, uint32_t size);

/**
 * @brief init crop config
 * @param [in] width: the width of pic that you want to crop
 * @param [in] height: the height of pic that you want to crop
 * @return acldvppRoiConfig
 **/
acldvppRoiConfig *InitCropRoiConfig(uint32_t width, uint32_t height);

/**
 * @brief init center crop config
 * @param [in] newInputWidth: width before crop
 * @param [in] newInputHeight: height before crop
 * @param [in] modelInputWidth: width after crop
 * @param [in] modelInputHeight: height after crop
 * @return acldvppRoiConfig
 **/
acldvppRoiConfig *InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight, uint32_t modelInputWidth, uint32_t modelInputHeight);

/**
 * @brief get the width and height of jpeg pic
 * @param [in] buff: ptr of pic memory address
 * @param [in] fileSize: size of pic memory
 * @param [in] fileLocation: location of pic
 * @param [out] W: widht of pic
 * @param [out] H: height of pic
 **/
void GetImageHW(void* buff, uint32_t fileSize, std::string fileLocation, uint32_t &W, uint32_t &H);

/**
 * @brief dvpp process of resnet50HC
 * @param [in] fileLocation: location of pic
 * @param [in] ptr: ptr of pic memory address on device
 * @return result
 **/
aclError DVPP_Resnet50HC(std::string fileLocation, char *&ptr);

#endif
