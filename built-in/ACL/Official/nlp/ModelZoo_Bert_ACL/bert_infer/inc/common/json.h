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

#ifndef _JSON_H_
#define _JSON_H_

#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "common.h"
// #include "hw_log.h"
#include <dirent.h>
#include <sstream>

#ifdef VERSION_CTRL_CPU_1951
#include "dvpp/HiDvpp.h"
#endif

#ifdef VERSION_CTRL_CPU_1910
#include "dvpp/Vpc.h"
#include "dvpp/Vdec.h"
#include "dvpp/DvppCommon.h"
#endif

#ifdef VERSION_HOST
#include "acl/ops/acl_dvpp.h"
enum JPEGD_TYPE {
    JPEGD_TYPE_JPEG = H264_HIGH_LEVEL + 1000,
    JPEGD_TYPE_MJPEG
};
#endif

using json = nlohmann::json;

#define VENC_SRC_RATE_MIN (1)
#define VENC_SRC_RATE_MAX (120)

#define VENC_MAXBIT_RATE_MIN (10)
#define VENC_MAXBIT_RATE_MAX (30000)

struct vpcAreaConfig {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;
};

struct dataConfig {
    uint32_t thread_num;
    std::vector<std::string> dir_path_vec;
    std::vector<std::string> img_info_vec;
    uint32_t batch_size;
    std::string image_format;
    uint32_t frame_rate;
    std::vector<uint32_t> img_width_vec;
    std::vector<uint32_t> img_height_vec;
#ifdef VERSION_CTRL_CPU_1951
    std::vector<PIXEL_FORMAT_E> img_format_vec;
#endif

#ifdef VERSION_CTRL_CPU_1910
    std::vector<VpcInputFormat> img_format_vec;
#endif

#ifdef VERSION_HOST
    std::vector<acldvppPixelFormat> img_format_vec;
#endif
};

struct VpcConfig {
    uint32_t vpc_channel_num;
#ifdef VERSION_CTRL_CPU_1951
    uint32_t s32_millisec;
    HI_U32 interpolation;
    double vpcFx;
    double vpcFy;
    PIXEL_FORMAT_E vpc_in_img_format;
    PIXEL_FORMAT_E vpc_out_img_format;
    VPC_MAKE_BORDER_INFO_S border_info;
#endif
#ifdef VERSION_CTRL_CPU_1910
    VpcInputFormat vpc_in_img_format;
    VpcOutputFormat vpc_out_img_format;
#endif
#ifdef VERSION_HOST
    acldvppPixelFormat vpc_in_img_format;
    acldvppPixelFormat vpc_out_img_format;
#endif
    uint32_t vpc_out_width;
    uint32_t vpc_out_height;
    uint32_t vpc_type;
    vpcAreaConfig crop_area;
    vpcAreaConfig paste_area;
    uint32_t batch_pic_num;
    uint32_t roi_num;
    uint32_t save_dvpp_file_flg;
    std::string save_dvpp_file_path;
};

// vdec & jpegd & png json config
struct VdecConfigParams {
    uint32_t channelNum;

    uint32_t width;
    uint32_t height;
    uint32_t outWidth;
    uint32_t outHeight;
    std::string inputFile;

    int enType;
    int enPixelFormat;
#ifdef VERSION_CTRL_CPU_1951
    uint32_t apha;
    uint32_t refFrameNum;
    uint32_t displayFrameNum;

    VIDEO_MODE_E enMode;
    COMPRESS_MODE_E enCompressMode;
    VIDEO_FORMAT_E enVedioFormat;
    VIDEO_OUTPUT_ORDER_E enOutputOrder;
    VIDEO_DEC_MODE_E enDecMode;
    DATA_BITWIDTH_E enBitWidth;

#endif

#ifdef VERSION_CTRL_CPU_1910
    std::string streamFormat;
#endif

    int32_t milliSec;
    int32_t intervalTime;
    std::string outputFormat;
    bool isVBeforeU;
};

// venc json config
struct VencConfigParams {
    uint32_t channelNum; // venc thread num

#ifdef VERSION_HOST
    acldvppPixelFormat picFormat; // input picture format
    acldvppStreamFormat enType;   // encode protocol
#endif

    uint32_t picWidth;         // input picture width
    uint32_t picHeight;        // input picture height
    uint32_t keyFrameInterval; // key frame interval, can not zero
    int rcMode;                // encode rate control mode, 1-VBR, 2-CBR, default CBR
    uint32_t srcRate;          // input stream rate, unit fps, scope[1, 120], default 30
    uint32_t maxBitRate;       // output code stream rate, scope[10,30000], default 300
    std::string outFolder;
    uint32_t outFileNumMax;
    uint32_t memPoolSize;
};

// cscé?óò×a??2?êy
struct aippCscConfig {
    int8_t csc_switch;
    int16_t cscMatrixR0C0;
    int16_t cscMatrixR0C1;
    int16_t cscMatrixR0C2;
    int16_t cscMatrixR1C0;
    int16_t cscMatrixR1C1;
    int16_t cscMatrixR1C2;
    int16_t cscMatrixR2C0;
    int16_t cscMatrixR2C1;
    int16_t cscMatrixR2C2;
    uint8_t cscOutputBiasR0;
    uint8_t cscOutputBiasR1;
    uint8_t cscOutputBiasR2;
    uint8_t cscInputBiasR0;
    uint8_t cscInputBiasR1;
    uint8_t cscInputBiasR2;
};

// ??·??à1?μ?????
struct aippScfConfig {
    int8_t scfSwitch;
    int32_t scfInputSizeW;
    int32_t scfInputSizeH;
    int32_t scfOutputSizeW;
    int32_t scfOutputSizeH;
    uint64_t batchIndex;
};

// ?ùí??à1?μ?2?êy
struct aippCropConfig {
    int8_t cropSwitch;
    int32_t cropStartPosW;
    int32_t cropStartPosH;
    int32_t cropSizeW;
    int32_t cropSizeH;
    uint64_t batchIndex;
};

// 21±??à1?μ?2?êy
struct aippPaddingConfig {
    int8_t paddingSwitch;
    int32_t paddingSizeTop;
    int32_t paddingSizeBottom;
    int32_t paddingSizeLeft;
    int32_t paddingSizeRight;
    uint64_t batchIndex;
};

// í¨μàμ??ù?μ?à1?2?êy
struct aippDtcPixelMeanConfig {
    int16_t dtcPixelMeanChn0;
    int16_t dtcPixelMeanChn1;
    int16_t dtcPixelMeanChn2;
    int16_t dtcPixelMeanChn3;
    uint64_t batchIndex;
};

// í¨μàμ?×?D??μ?à1?2?êy
struct aippDtcPixelMinConfig {
    float dtcPixelMinChn0;
    float dtcPixelMinChn1;
    float dtcPixelMinChn2;
    float dtcPixelMinChn3;
    uint64_t batchIndex;
};

// í¨μàμ?·?2??à1?2?êy
struct aippPixelVarReciConfig {
    float dtcPixelVarReciChn0;
    float dtcPixelVarReciChn1;
    float dtcPixelVarReciChn2;
    float dtcPixelVarReciChn3;
    uint64_t batchIndex;
};

struct dynamicImgConfig {
    size_t shapeH;
    size_t shapeW;
};

struct dynamicDimsConfig {
    aclmdlIODims dydims;
};

struct dynamic_aipp_config {
    aclAippInputFormat inputFormat;
    int32_t srcImageSizeW;
    int32_t srcImageSizeH;
    aippCscConfig cscParams;
    int8_t rbuvSwapSwitch;
    int8_t axSwapSwitch;

    uint32_t scfCfgNum;
    aippScfConfig *scfParams;

    uint32_t cropCfgNum;
    aippCropConfig *cropParams;

    uint32_t padCfgNum;
    aippPaddingConfig *paddingParams;

    uint32_t dtcPixelMeanCfgNum;
    aippDtcPixelMeanConfig *dtcPixelMeanParams;

    uint32_t dtcPixelMinCfgNum;
    aippDtcPixelMinConfig *dtcPixelMinParams;

    uint32_t pixelVarReciCfgNum;
    aippPixelVarReciConfig *pixelVarReciParams;
};

struct common_config {
    uint32_t device_num;
    std::vector<uint32_t> device_id_vec;
    uint32_t loopNum;
    std::string frame_work;
    uint8_t inferFlag;
    uint8_t vdecFlag;
    uint8_t jpegdFlag;
    uint8_t pngdFlag;
    uint8_t vpcFlag;
    uint8_t vencFlag;
};

struct infer_asyn_parameter {
    uint32_t mem_pool_size;
    uint32_t callback_interval;
};

typedef enum {
    inference_type_syn,
    inference_type_asyn_block,
    inference_type_asyn_noblock,
    inference_type_count
} Infer_Type;

struct inference_config {
    std::string imgType;
    uint64_t batch_size;
    uint32_t infer_type;
    uint32_t channelNum;
    uint32_t postType;
    uint32_t modelNum;
    std::vector<std::string> modelType;
    std::vector<std::string> omPatch;
    std::vector<std::string> inputFilePath;
    std::vector<std::string> resultFolderPath;
    uint8_t dynamicBathFlag;
    uint8_t dynamicImgFlag;
    uint8_t dynamicAippFlag;
    // xwx5322041
    uint8_t dynamicDimsFlag;

    std::string resnetStdFile;
    std::string yoloImgInfoFile;

    infer_asyn_parameter inferAsynPara;
    dynamicImgConfig dynamicImg;
    dynamic_aipp_config dynamicAippCfg;
    // xwx5322041
    dynamicDimsConfig dynamicDims;
};

struct inferenceJsonConfig {
    std::string test_case_name; // ó?ày??3?
    std::string test_scenario;  // ó?ày??±?
    common_config commCfg;
    dataConfig dataCfg;
    inference_config inferCfg;

    VdecConfigParams vdecCfgPara;
    VpcConfig vpcCfg;
    VencConfigParams vencCfg;
};


extern std::string testcaseName;
extern std::string test_scenario;

extern std::map<std::string, getJsonConfig_t> g_testScenarioString2int;

extern inferenceJsonConfig inference_json_cfg_tbl;

int getConfigFromJsonFile(char *jsonpath);

void getInferenceConfigFromJson(char *jsonpath, void *config);

#endif