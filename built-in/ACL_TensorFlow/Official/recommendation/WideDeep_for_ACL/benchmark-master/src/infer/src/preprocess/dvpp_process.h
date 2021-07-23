/**
* @file dvpp_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef DVPP_PREPROCESS_H
#define DVPP_PREPROCESS_H
//#pragma once
#include <cstdint>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <string>
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO][Vision]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN][Vision]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR][Vision] " fmt "\n", ##args)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct PicDesc {
    std::string picName;
    int width;
    int height;
} PicDesc;

/**
 * DvppProcess
 */
class DvppProcess {
public:
    /**
    * @brief Constructor
    * @param [in] stream: stream
    */
    DvppProcess(aclrtStream &stream);

    /**
    * @brief Destructor
    */
    ~DvppProcess();

    /**
    * @brief dvpp global init
    * @return result
    */
    Result InitResource();

    /**
    * @brief init dvpp output para
    * @param [in] modelInputWidth: model input width
    * @param [in] modelInputHeight: model input height
    * @return result
    */
    Result InitOutputPara(int modelInputWidth, int modelInputHeight);

    /**
    * @brief set dvpp input
    * @param [in] inDevBuffer: device buffer of input pic
    * @param [in] inDevBufferSize: device buffer size of input pic
    * @param [in] inputWidth:width of pic
    * @param [in] inputHeight:height of pic
    */
    void SetInput(void *inDevBuffer, int inDevBufferSize, int inputWidth, int inputHeight);

    /**
    * @brief gett dvpp output
    * @param [in] outputBuffer: pointer which points to dvpp output buffer
    * @param [out] outputSize: output size
    */
    void GetOutput(void **outputBuffer, int &outputSize);

    /**
    * @brief dvpp process
    * @return result
    */
    Result Process();

private:
    Result InitDecodeOutputDesc();
    Result ProcessDecode();
    void DestroyDecodeResource();

    Result InitResizeInputDesc();
    Result InitResizeOutputDesc();
    Result ProcessResize();
    void DestroyResizeResource();

    void DestroyResource();

    void DestroyOutputPara();

    acldvppChannelDesc *dvppChannelDesc_;
    aclrtStream stream_;
    acldvppResizeConfig *resizeConfig_;

    void *decodeOutDevBuffer_;         // decode output buffer
    acldvppPicDesc *decodeOutputDesc_; //decode output desc

    acldvppPicDesc *resizeInputDesc_;  // resize input desc
    acldvppPicDesc *resizeOutputDesc_; // resize output desc

    void *inDevBuffer_;        // decode input buffer
    uint32_t inDevBufferSize_; // dvpp input buffer size

    uint32_t inputWidth_;  // input pic width
    uint32_t inputHeight_; // input pic height

    void *resizeOutBufferDev_;     // resize output buffer
    uint32_t resizeOutBufferSize_; // resize output size

    uint32_t modelInputWidth_;  // model input width
    uint32_t modelInputHeight_; // model input height
};

#endif
