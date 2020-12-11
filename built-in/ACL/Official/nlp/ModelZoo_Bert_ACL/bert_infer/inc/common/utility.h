/* *
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <getopt.h>
#include "common.h"
#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"
#include "acl/acl_base.h"


int SdkInferScanFiles(std::vector<std::string> &fileList,
    std::string inputDirectory);                                     // Obtain the file name and number of files
char *SdkInferReadBinFile(std::string fileName, uint32_t &fileSize); // Reading the bin file
aclError SdkInferWriteToFile(FILE *fileFp, void *dataDev, uint32_t dataSize, bool isDevice);

#define SEND_END_FRAME(Queue)                                        \
    shared_ptr<Trans_Buff_T> endFrame = make_shared<Trans_Buff_T>(); \
    endFrame->endFlag = true;                                        \
    (Queue).put(endFrame);

// Global Variable Definition
typedef struct Trans_Buff {
    // 消息的功能类型
    uint64_t type_id;
    uint32_t img_height;
    uint32_t img_width;
    uint32_t components;
    uint32_t img_width_aligned;
    uint32_t img_height_aligned;
    uint32_t decode_out_format;

    uint32_t mdl_height;
    uint32_t mdl_width;

    std::shared_ptr<void> img_addr;
    uint64_t imgBuf_size;
    // 被传递的buff指针: image
    std::shared_ptr<void> trans_buff;
    // 被传递的buff大小: image size
    uint64_t buffer_size;

    // frame 编号
    uint64_t frame_id;  // frame index
    uint32_t frame_num; // total frame number
    std::string img_name;
    bool endFlag;
    std::vector<std::string> imgName_vec;
    // 模型推理使用的数据
    aclmdlDataset *model_input_data;
    aclmdlDataset *model_output_data;
    DetBox detBoxInfo;
} Trans_Buff_T;

typedef enum Trans_Buff_Type {
    TYPE_VDEC,
    TYPE_JPEGD,
    TYPE_PNGD,
    TYPE_VPC,
    TYPE_NN,
    TYPE_YUV,
    TYPE_END // 数据结束(用于线程队列的退出使用)
} Trans_Buff_Type_T;

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;


typedef enum DvppType {
    VpcResize = 0,
    VpcCrop = 1,
    VpcCropAndPaste = 2,
    JpegE = 3
} DvppType;

typedef struct PicDesc {
    std::string picName;
    int width;
    int height;
} PicDesc;
class RunStatus {
public:
    static void SetDeviceStatus(bool isDevice)
    {
        isDevice_ = isDevice;
    }
    static bool GetDeviceStatus()
    {
        return isDevice_;
    }

private:
    RunStatus() {}
    ~RunStatus() {}
    static bool isDevice_;
};

class SdkInferUtils {
public:
    /* *
     * @brief create device buffer of pic
     * @param [in] picDesc: pic desc
     * @param [out] devPicBufferSize: size of pic
     * @return device buffer of pic
     */
    static void *GetDeviceBufferOfPicture(const PicDesc &picDesc, uint32_t &devPicBufferSize);
    static void *ReadBinFile1(std::string fileName, uint32_t &fileSize);

    /* *
     * @brief create buffer of file
     * @param [in] fileName: file name
     * @param [out] fileSize: size of file
     * @return buffer of pic
     */
    static void *ReadBinFile2(std::string fileName, uint32_t &fileSize);

    /* *
     * @brief create device buffer of pic
     * @param [in] picDesc: pic desc
     * @param [out] devPicBufferSize: actual pic size
     * @return device buffer of pic
     */
    static char *GetPicDevBuffer4JpegD(const PicDesc &picDesc, uint32_t &devPicBufferSize);

    /* *
     * @brief create device buffer of pic
     * @param [in] picDesc: pic desc
     * @param [in] PicBufferSize: aligned pic size
     * @return device buffer of pic
     */
    static char *GetPicDevBuffer4JpegE(const PicDesc &picDesc, uint32_t &PicBufferSize);

    /* *
     * @brief save dvpp output data
     * @param [in] fileName: file name
     * @param [in] devPtr: dvpp output data device addr
     * @param [in] dataSize: dvpp output data size
     * @return result
     */
    static Result SaveDvppOutputData(const char *fileName, void *devPtr, uint32_t dataSize);

    /* *
     * @brief check file if exist
     * @param [in] fileName: file to check
     * @return result
     */
    static Result CheckFile(const char *fileName);
    /* *
     * @brief save model output data to dst file
     * @param [in] srcfileName: src file name
     * @param [in] dstfileName: dst file name
     * @return result
     */
    static Result SaveModelOutputData(const char *srcfileName, const char *dstfileName);

    static void *GetDeviceBufferOfFile(std::string fileName, uint32_t &fileSize);

    /* *
     * @brief create buffer of file
     * @param [in] fileName: file name
     * @param [out] fileSize: size of file
     * @return buffer of pic
     */
    static void *ReadBinFile(std::string fileName, uint32_t &fileSize);
};

#endif