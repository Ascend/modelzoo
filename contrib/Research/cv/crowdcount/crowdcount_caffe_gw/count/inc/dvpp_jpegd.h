/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File dvpp_process.h
* Description: handle dvpp process
*/
#pragma once
#include <cstdint>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "utils.h"


/**
 * DvppProcess
 */
class DvppJpegD {
public:
    /**
    * @brief Constructor
    * @param [in] stream: stream
    */
    DvppJpegD(aclrtStream &stream,  acldvppChannelDesc *dvppChannelDesc);

    /**
    * @brief Destructor
    */
    ~DvppJpegD();

    /**
    * @brief dvpp global init
    * @return result
    */
    Result init_resource();

    /**
    * @brief init dvpp output para
    * @param [in] modelInputWidth: model input width
    * @param [in] modelInputHeight: model input height
    * @return result
    */
    Result init_output_para(int modelInputWidth, int modelInputHeight);

    /**
    * @brief set jpegd input
    * @param [in] inDevBuffer: device buffer of input pic
    * @param [in] inDevBufferSize: device buffer size of input pic
    * @param [in] inputWidth:width of pic
    * @param [in] inputHeight:height of pic
    */
    void set_input_jpeg_d(uint8_t* inDevBuffer, int inDevBufferSize, int inputWidth, int inputHeight);
    Result init_decode_output_desc(ImageData& inputImage);
    /**
    * @brief gett dvpp output
    * @param [in] outputBuffer: pointer which points to dvpp output buffer
    * @param [out] outputSize: output size
    */
    void get_output(void **outputBuffer, int &outputSize);
    Result process(ImageData& dest, ImageData& src);
   /**
    * @brief release encode resource
    */
    void destroy_encode_resource();

private:
    void destroy_decode_resource();
    void destroy_resource();
    void destroy_outputPara();

    aclrtStream stream_;
    acldvppChannelDesc *dvppChannelDesc_;


    void* decodeOutBufferDev_; // decode output buffer
    acldvppPicDesc *decodeOutputDesc_; //decode output desc


    uint8_t *inDevBuffer_;  // input pic dev buffer
    uint32_t inDevBufferSizeD_; // input pic size for decode

    void *vpcOutBufferDev_; // vpc output buffer
    uint32_t vpcOutBufferSize_;  // vpc output size
};

