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

#include "json.h"


std::string testcaseName;
std::string test_scenario;

std::map<std::string, aclAippInputFormat> g_inputFormat_map = {
    { "ACL_YUV420SP_U8", ACL_YUV420SP_U8 },
    { "ACL_XRGB8888_U8", ACL_XRGB8888_U8 },
    { "ACL_RGB888_U8", ACL_RGB888_U8 },
    { "ACL_YUV400_U8", ACL_YUV400_U8 },
    { "ACL_NC1HWC0DI_FP16", ACL_NC1HWC0DI_FP16 },
    { "ACL_NC1HWC0DI_S8", ACL_NC1HWC0DI_S8 },
    { "ACL_ARGB8888_U8", ACL_ARGB8888_U8 },
    { "ACL_YUYV_U8", ACL_YUYV_U8 },
    { "ACL_YUV422SP_U8", ACL_YUV422SP_U8 },
    { "ACL_AYUV444_U8 ", ACL_AYUV444_U8 },
    { "ACL_RAW10", ACL_RAW10 },
    { "ACL_RAW12", ACL_RAW12 },
    { "ACL_RAW16", ACL_RAW16 },
    { "ACL_RAW24 ", ACL_RAW24 },
};

std::map<std::string, int> g_rcMode2enum_map = {
    { "VBR", 1 },
    { "CBR", 2 },
};

#ifdef VERSION_CTRL_CPU_1951
std::map<std::string, PIXEL_FORMAT_E> g_1951devDvppInputFormat_map = {
    { "YUV400", PIXEL_FORMAT_YUV_400 },
    { "YUV420SP", PIXEL_FORMAT_YUV_SEMIPLANAR_420 },
    { "YVU420SP", PIXEL_FORMAT_YVU_SEMIPLANAR_420 },
    { "YUV422SP", PIXEL_FORMAT_YUV_SEMIPLANAR_422 },
    { "YVU422SP", PIXEL_FORMAT_YVU_SEMIPLANAR_422 },
    { "YUV444SP", PIXEL_FORMAT_YUV_SEMIPLANAR_444 },
    { "YVU444SP", PIXEL_FORMAT_YVU_SEMIPLANAR_444 },
    { "YUYV422P", PIXEL_FORMAT_YUYV_PACKED_422 },
    { "UYVY422P", PIXEL_FORMAT_UYVY_PACKED_422 },
    { "YVYU422P", PIXEL_FORMAT_YVYU_PACKED_422 },
    { "VYUY422P", PIXEL_FORMAT_VYUY_PACKED_422 },
    { "YUV444P", PIXEL_FORMAT_YUV_PACKED_444 },
    { "RGB888", PIXEL_FORMAT_RGB_888 },
    { "BGR888", PIXEL_FORMAT_BGR_888 },
    { "ARGB8888", PIXEL_FORMAT_ARGB_8888 },
    { "ABGR8888", PIXEL_FORMAT_ABGR_8888 },
    { "RGBA8888", PIXEL_FORMAT_RGBA_8888 },
    { "BGRA8888", PIXEL_FORMAT_BGRA_8888 },
    { "YUV440SP", PIXEL_FORMAT_YUV_SEMIPLANAR_440 },
    { "YVU440SP", PIXEL_FORMAT_YVU_SEMIPLANAR_440 },
};
std::map<std::string, VPC_BORD_TYPE_E> g_1951devBordType_map = {
    { "BORDER_CONSTANT", BORDER_CONSTANT },
    { "BORDER_REPLICATE", BORDER_REPLICATE },
    { "BORDER_REFLECT", BORDER_REFLECT },
    { "BORDER_REFLECT_101", BORDER_REFLECT_101 },
};

std::map<std::string, PAYLOAD_TYPE_E> g_payloadType2enum_map = {
    { "PT_H264", PT_H264 },
    { "PT_H265", PT_H265 },
    { "PT_JPEG", PT_JPEG },
    { "PT_MJPEG", PT_MJPEG },
};

std::map<std::string, VIDEO_MODE_E> g_mode2enum_map = {
    { "VIDEO_MODE_STREAM", VIDEO_MODE_STREAM },
    { "VIDEO_MODE_FRAME", VIDEO_MODE_FRAME },
    { "VIDEO_MODE_COMPAT", VIDEO_MODE_COMPAT },
    { "VIDEO_MODE_BUTT", VIDEO_MODE_BUTT },
};

std::map<std::string, VIDEO_DEC_MODE_E> g_decMode2enum_map = {
    { "VIDEO_DEC_MODE_IPB", VIDEO_DEC_MODE_IPB },
    { "VIDEO_DEC_MODE_IP", VIDEO_DEC_MODE_IP },
    { "VIDEO_DEC_MODE_I", VIDEO_DEC_MODE_I },
    { "VIDEO_DEC_MODE_BUTT", VIDEO_DEC_MODE_BUTT },
};

std::map<std::string, DATA_BITWIDTH_E> g_bitWidth2enum_map = {
    { "DATA_BITWIDTH_8", DATA_BITWIDTH_8 },   { "DATA_BITWIDTH_10", DATA_BITWIDTH_10 },
    { "DATA_BITWIDTH_12", DATA_BITWIDTH_12 }, { "DATA_BITWIDTH_14", DATA_BITWIDTH_14 },
    { "DATA_BITWIDTH_16", DATA_BITWIDTH_16 }, { "DATA_BITWIDTH_BUTT", DATA_BITWIDTH_BUTT },
};

std::map<std::string, PIXEL_FORMAT_E> g_pixelFormat2enum_map = {
    { "PIXEL_FORMAT_YUV_400", PIXEL_FORMAT_YUV_400 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_420", PIXEL_FORMAT_YUV_SEMIPLANAR_420 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_420", PIXEL_FORMAT_YVU_SEMIPLANAR_420 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_422", PIXEL_FORMAT_YUV_SEMIPLANAR_422 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_422", PIXEL_FORMAT_YVU_SEMIPLANAR_422 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_444", PIXEL_FORMAT_YUV_SEMIPLANAR_444 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_444", PIXEL_FORMAT_YVU_SEMIPLANAR_444 },

    { "PIXEL_FORMAT_YUYV_PACKED_422", PIXEL_FORMAT_YUYV_PACKED_422 },
    { "PIXEL_FORMAT_UYVY_PACKED_422", PIXEL_FORMAT_UYVY_PACKED_422 },
    { "PIXEL_FORMAT_YVYU_PACKED_422", PIXEL_FORMAT_YVYU_PACKED_422 },
    { "PIXEL_FORMAT_VYUY_PACKED_422", PIXEL_FORMAT_VYUY_PACKED_422 },
    { "PIXEL_FORMAT_YUV_PACKED_444", PIXEL_FORMAT_YUV_PACKED_444 },
    { "PIXEL_FORMAT_RGB_888", PIXEL_FORMAT_RGB_888 },
    { "PIXEL_FORMAT_BGR_888", PIXEL_FORMAT_BGR_888 },
    { "PIXEL_FORMAT_ARGB_8888", PIXEL_FORMAT_ARGB_8888 },
    { "PIXEL_FORMAT_ABGR_8888", PIXEL_FORMAT_ABGR_8888 },
    { "PIXEL_FORMAT_RGBA_8888", PIXEL_FORMAT_RGBA_8888 },
    { "PIXEL_FORMAT_BGRA_8888", PIXEL_FORMAT_BGRA_8888 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_440", PIXEL_FORMAT_YUV_SEMIPLANAR_440 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_440", PIXEL_FORMAT_YVU_SEMIPLANAR_440 },
};

std::map<std::string, COMPRESS_MODE_E> g_compressMode2enum_map = {
    { "COMPRESS_MODE_NONE", COMPRESS_MODE_NONE },
    { "COMPRESS_MODE_SEG", COMPRESS_MODE_SEG },
    { "COMPRESS_MODE_TILE", COMPRESS_MODE_TILE },
    { "COMPRESS_MODE_HFBC", COMPRESS_MODE_HFBC },
};

std::map<std::string, VIDEO_FORMAT_E> g_vedioFormat2enum_map = {
    { "VIDEO_FORMAT_LINEAR", VIDEO_FORMAT_LINEAR },
    { "VIDEO_FORMAT_TILE_64x16", VIDEO_FORMAT_TILE_64x16 },
};

std::map<std::string, VIDEO_OUTPUT_ORDER_E> g_outputOrder2enum_map = {
    { "VIDEO_OUTPUT_ORDER_DISP", VIDEO_OUTPUT_ORDER_DISP },
    { "VIDEO_OUTPUT_ORDER_DEC", VIDEO_OUTPUT_ORDER_DEC },
};

#endif

#ifdef VERSION_CTRL_CPU_1910
std::map<std::string, VpcInputFormat> g_1910devDvppInputFormat_map = {
    { "YUV400", INPUT_YUV400 },
    { "YUV420SP", INPUT_YUV420_SEMI_PLANNER_UV },
    { "YVU420SP", INPUT_YUV420_SEMI_PLANNER_VU },
    { "YUV422SP", INPUT_YUV422_SEMI_PLANNER_UV },
    { "YVU422SP", INPUT_YUV422_SEMI_PLANNER_VU },
    { "YUV444SP", INPUT_YUV444_SEMI_PLANNER_UV },
    { "YVU444SP", INPUT_YUV444_SEMI_PLANNER_VU },
    { "YUYV422P", INPUT_YUV422_PACKED_YUYV },
    { "UYVY422P", INPUT_YUV422_PACKED_UYVY },
    { "YVYU422P", INPUT_YUV422_PACKED_YVYU },
    { "VYUY422P", INPUT_YUV422_PACKED_VYUY },
    { "YUV444P", INPUT_YUV444_PACKED_YUV },
    { "RGB888", INPUT_RGB },
    { "BGR888", INPUT_BGR },
    { "ARGB8888", INPUT_ARGB },
    { "ABGR8888", INPUT_ABGR },
    { "RGBA8888", INPUT_RGBA },
    { "BGRA8888", INPUT_BGRA },
    { "YUV440SP_U10", INPUT_YUV420_SEMI_PLANNER_UV_10BIT },
    { "YVU440SP_U10", INPUT_YUV420_SEMI_PLANNER_VU_10BIT },
};
std::map<std::string, VpcOutputFormat> g_1910devDvppOutFormat_map = {
    { "YUV420SP", OUTPUT_YUV420SP_UV },
    { "YVU420SP", OUTPUT_YUV420SP_VU },
};

std::map<std::string, std::string> g_payloadType2Name_map = {
    { "PT_H264", "h264" },
    { "PT_H265", "h265" },
};


#endif

#ifdef VERSION_HOST
std::map<std::string, acldvppPixelFormat> g_aclDvppInputFormat_map = {
    { "YUV400", PIXEL_FORMAT_YUV_400 },
    { "YUV420SP", PIXEL_FORMAT_YUV_SEMIPLANAR_420 },
    { "YVU420SP", PIXEL_FORMAT_YVU_SEMIPLANAR_420 },
    { "YUV422SP", PIXEL_FORMAT_YUV_SEMIPLANAR_422 },
    { "YVU422SP", PIXEL_FORMAT_YVU_SEMIPLANAR_422 },
    { "YUV444SP", PIXEL_FORMAT_YUV_SEMIPLANAR_444 },
    { "YVU444SP", PIXEL_FORMAT_YVU_SEMIPLANAR_444 },
    { "YUYV422P", PIXEL_FORMAT_YUYV_PACKED_422 },
    { "UYVY422P", PIXEL_FORMAT_UYVY_PACKED_422 },
    { "YVYU422P", PIXEL_FORMAT_YVYU_PACKED_422 },
    { "VYUY422P", PIXEL_FORMAT_VYUY_PACKED_422 },
    { "YUV444P", PIXEL_FORMAT_YUV_PACKED_444 },
    { "RGB888", PIXEL_FORMAT_RGB_888 },
    { "BGR888", PIXEL_FORMAT_BGR_888 },
    { "ARGB8888", PIXEL_FORMAT_ARGB_8888 },
    { "ABGR8888", PIXEL_FORMAT_ABGR_8888 },
    { "RGBA8888", PIXEL_FORMAT_RGBA_8888 },
    { "BGRA8888", PIXEL_FORMAT_BGRA_8888 },
    { "UNKNOWN", PIXEL_FORMAT_UNKNOWN },
    { "YUV440SP", PIXEL_FORMAT_YUV_SEMIPLANAR_440 },
    { "YVU440SP", PIXEL_FORMAT_YVU_SEMIPLANAR_440 },
};

std::map<std::string, acldvppStreamFormat> g_type2enum_map = {
    { "PT_H264", H264_MAIN_LEVEL },
    { "PT_H265", H265_MAIN_LEVEL },
    { "PT_H264_BASE", H264_BASELINE_LEVEL },
    { "PT_H264_HIGH", H264_HIGH_LEVEL },
};

std::map<std::string, JPEGD_TYPE> g_jpegdType_map = {
    { "PT_JPEG", JPEGD_TYPE_JPEG },
    { "PT_MJPEG", JPEGD_TYPE_MJPEG },
};

std::map<std::string, acldvppPixelFormat> g_pixelFormat2enum_map = {
    { "PIXEL_FORMAT_YUV_400", PIXEL_FORMAT_YUV_400 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_420", PIXEL_FORMAT_YUV_SEMIPLANAR_420 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_420", PIXEL_FORMAT_YVU_SEMIPLANAR_420 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_422", PIXEL_FORMAT_YUV_SEMIPLANAR_422 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_422", PIXEL_FORMAT_YVU_SEMIPLANAR_422 },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_444", PIXEL_FORMAT_YUV_SEMIPLANAR_444 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_444", PIXEL_FORMAT_YVU_SEMIPLANAR_444 },

    { "PIXEL_FORMAT_YUYV_PACKED_422", PIXEL_FORMAT_YUYV_PACKED_422 },
    { "PIXEL_FORMAT_UYVY_PACKED_422", PIXEL_FORMAT_UYVY_PACKED_422 },
    { "PIXEL_FORMAT_YVYU_PACKED_422", PIXEL_FORMAT_YVYU_PACKED_422 },
    { "PIXEL_FORMAT_VYUY_PACKED_422", PIXEL_FORMAT_VYUY_PACKED_422 },
    { "PIXEL_FORMAT_YUV_PACKED_444", PIXEL_FORMAT_YUV_PACKED_444 },
    { "PIXEL_FORMAT_RGB_888", PIXEL_FORMAT_RGB_888 },
    { "PIXEL_FORMAT_BGR_888", PIXEL_FORMAT_BGR_888 },
    { "PIXEL_FORMAT_ARGB_8888", PIXEL_FORMAT_ARGB_8888 },
    { "PIXEL_FORMAT_ABGR_8888", PIXEL_FORMAT_ABGR_8888 },
    { "PIXEL_FORMAT_RGBA_8888", PIXEL_FORMAT_RGBA_8888 },
    { "PIXEL_FORMAT_BGRA_8888", PIXEL_FORMAT_BGRA_8888 },
    { "PIXEL_FORMAT_UNKNOWN", PIXEL_FORMAT_UNKNOWN },
    { "PIXEL_FORMAT_YUV_SEMIPLANAR_440", PIXEL_FORMAT_YUV_SEMIPLANAR_440 },
    { "PIXEL_FORMAT_YVU_SEMIPLANAR_440", PIXEL_FORMAT_YVU_SEMIPLANAR_440 },
};

#endif
typedef enum VPCTYPE {
    VPC_RESIZE = 0,
    VPC_CROP = 1,
    VPC_CROP_AND_PASTE,
    VPC_BATCH_CROP,
    VPC_BATCH_CROP_AND_PASTE,
    VPC_CROP_RESIZE,
    VPC_CROP_RESIZE_PASTE,
    CONVERT_COLOR,
    CONVERT_COLOR2YUV420,
    VPC_COPY_MAKE_BORDER,
    VPC_PYR_DOWN,
    VPC_CROP_RESIZE_MAKE_BORDER
} VPCTYPE;

std::map<std::string, uint32_t> g_vpcType = {
    { "resize", VPC_RESIZE },
    { "crop", VPC_CROP },
    { "cropAndPaste", VPC_CROP_AND_PASTE },
    { "batchCrop", VPC_BATCH_CROP },
    { "batchCropAndPaste", VPC_BATCH_CROP_AND_PASTE },
    { "cropResize", VPC_CROP_RESIZE },
    { "cropResizePaste", VPC_CROP_RESIZE_PASTE },
    { "convertColor", CONVERT_COLOR },
    { "convertColor2Yuv420", CONVERT_COLOR2YUV420 },
    { "copyMakeBorder", VPC_COPY_MAKE_BORDER },
    { "pyrDown", VPC_PYR_DOWN },
    { "copyResizemakeBorder", VPC_CROP_RESIZE_MAKE_BORDER },
};

inferenceJsonConfig inference_json_cfg_tbl;

getJsonConfig_t g_inferGetJson = { getInferenceConfigFromJson, (void *)(&inference_json_cfg_tbl) };

std::map<std::string, getJsonConfig_t> g_testScenario2jsonParse = { { "INFERENCE", g_inferGetJson } };

uint32_t get_intvalue(json &j, const char *nm, uint32_t defaultVal)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return uint32_t(j.at(nm));
    }
}

uint32_t get_intArrayvalue(json &j, const char *nm, uint32_t defaultVal, int i)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return uint32_t(j.at(nm)[i]);
    }
}

double get_doublevalue(json &j, const char *nm, double defaultVal)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return double(j.at(nm));
    }
}

double get_doubleArrayvalue(json &j, const char *nm, double defaultVal, int i)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return double(j.at(nm)[i]);
    }
}

std::string get_strvalue(json &j, const char *nm, std::string defaultVal)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return std::string(j.at(nm));
    }
}
std::string get_strArrayvalue(json &j, const char *nm, std::string defaultVal, int i)
{
    if (j.find(nm) == j.end()) {
        return defaultVal;
    } else {
        return std::string(j.at(nm)[i]);
    }
}

int getConfigFromJsonFile(char *jsonpath)
{
    // 读取json文件
    std::ifstream fromFile;
    fromFile.open(jsonpath);
    if (!fromFile.is_open()) {
        LOG_ERROR("open json file %s fail, please check", jsonpath);
        return -1;
    }

    json in;
    fromFile >> in;
    fromFile.close();

    // 读取根节点信息
    testcaseName = in["test_case_name"];
    LOG_INFO("testcaseName[%s]", testcaseName.c_str());
    test_scenario = in["test_scenario"];
    LOG_INFO("test_scenario[%s]", test_scenario.c_str());

    std::map<std::string, getJsonConfig_t>::iterator iter = g_testScenario2jsonParse.find(test_scenario);
    if (iter == g_testScenario2jsonParse.end()) {
        LOG_INFO("not found test scenario[%s] in map g_testScenario2jsonParse", test_scenario.c_str());
        return -1;
    }

    iter->second.callback_func(jsonpath, iter->second.p_config);

    return 0;
}

void getInferenceConfigFromJson(char *jsonpath, void *config)
{
    char cmd[256] = {0};
    std::ifstream fromFile;
    fromFile.open(jsonpath);
    if (!fromFile.is_open()) {
        LOG_ERROR("open json file [%s] fail", jsonpath);
        return;
    }

    json in;
    fromFile >> in;
    fromFile.close();
    inferenceJsonConfig *cfg = static_cast<inferenceJsonConfig *>(config);

    cfg->commCfg.device_num = get_intvalue(in["common"], "device_num", 1);
    LOG_INFO("device_num[%d]", cfg->commCfg.device_num);

    for (int i = 0; i < cfg->commCfg.device_num; i++) {
        uint32_t device_id = get_intArrayvalue(in["common"], "device_id_list", 0, i);
        cfg->commCfg.device_id_vec.push_back(device_id);
        LOG_INFO("cfg.device_id_vec[%d]=%d", i, cfg->commCfg.device_id_vec[i]);
    }

    cfg->commCfg.loopNum = get_intvalue(in["common"], "loop_num", 1);
    LOG_INFO("loop_num[%d]", cfg->commCfg.loopNum);

    cfg->commCfg.frame_work = get_strvalue(in["common"], "frame_work", "caffe");
    LOG_INFO("frame_work[%s]", cfg->commCfg.frame_work.c_str());

    cfg->commCfg.inferFlag = get_intvalue(in["common"], "infer_flag", 1);
    LOG_INFO("inferFlag[%d]", cfg->commCfg.inferFlag);

    cfg->commCfg.vdecFlag = get_intvalue(in["common"], "vdec_flag", 0);
    LOG_INFO("vdecFlag[%d]", cfg->commCfg.vdecFlag);

    cfg->commCfg.jpegdFlag = get_intvalue(in["common"], "jpegd_flag", 0);
    LOG_INFO("jpegdFlag[%d]", cfg->commCfg.jpegdFlag);

    cfg->commCfg.pngdFlag = get_intvalue(in["common"], "pngd_flag", 0);
    LOG_INFO("pngdFlag[%d]", cfg->commCfg.pngdFlag);

    cfg->commCfg.vpcFlag = get_intvalue(in["common"], "vpc_flag", 0);
    LOG_INFO("vpcFlag[%d]", cfg->commCfg.vpcFlag);

    cfg->commCfg.vencFlag = get_intvalue(in["common"], "venc_flag", 0);
    LOG_INFO("vencFlag[%d]", cfg->commCfg.vencFlag);

    cfg->dataCfg.thread_num = get_intvalue(in["dataset_config"], "dataset_channel_num", 0);
    for (int i = 0; i < in["dataset_config"]["dir_path_list"].size(); i++) {
        std::string dir_path =
            get_strArrayvalue(in["dataset_config"], "dir_path_list", "../datasets/ImageNet1024_224_224_YUV_bin", i);
        LOG_INFO("dir_path[%s]", dir_path.c_str());
        cfg->dataCfg.dir_path_vec.push_back(dir_path);
    }

    cfg->dataCfg.frame_rate = get_intvalue(in["dataset_config"], "frame_rate", 0);
    LOG_INFO("frame_rate[%u]", cfg->dataCfg.frame_rate);

    for (int i = 0; i < in["dataset_config"]["input_width"].size(); i++) {
        uint32_t width = get_intArrayvalue(in["dataset_config"], "input_width", 4096, i);
        cfg->dataCfg.img_width_vec.push_back(width);
        LOG_INFO("width[%d]", width);
    }
    for (int i = 0; i < in["dataset_config"]["input_height"].size(); i++) {
        uint32_t height = get_intArrayvalue(in["dataset_config"], "input_height", 4096, i);
        cfg->dataCfg.img_height_vec.push_back(height);
        LOG_INFO("height[%d]", height);
    }
    for (int i = 0; i < in["dataset_config"]["image_format"].size(); i++) {
        std::string tmp = get_strArrayvalue(in["dataset_config"], "image_format", "YUV420SP", i);

#ifdef VERSION_CTRL_CPU_1951
        PIXEL_FORMAT_E format = g_1951devDvppInputFormat_map[tmp];
        cfg->dataCfg.img_format_vec.push_back(format);
#endif

#ifdef VERSION_CTRL_CPU_1910
        VpcInputFormat format = g_1910devDvppInputFormat_map[tmp];
        cfg->dataCfg.img_format_vec.push_back(format);
#endif

#ifdef VERSION_HOST
        acldvppPixelFormat format = g_aclDvppInputFormat_map[tmp];
        cfg->dataCfg.img_format_vec.push_back(format);
#endif
    }

    cfg->dataCfg.image_format = get_strvalue(in["dataset_config"], "img_format", "jpg");

    if (cfg->commCfg.vencFlag == 1) {
        cfg->vencCfg.channelNum = get_intvalue(in["venc_config"], "channel_num", 1);

#ifdef VERSION_HOST
        std::string picFormatStr = get_strvalue(in["venc_config"], "pixel_format", "PIXEL_FORMAT_YUV_SEMIPLANAR_420");
        std::map<std::string, acldvppPixelFormat>::iterator picFormat_iter = g_pixelFormat2enum_map.find(picFormatStr);
        if (picFormat_iter == g_pixelFormat2enum_map.end()) {
            LOG_ERROR("[INFO]not found enType[%s] in map g_payloadType2enum_map", picFormatStr.c_str());
        }
        cfg->vencCfg.picFormat = picFormat_iter->second;
        LOG_INFO("vencCfg.picFormat[%d]", (int)cfg->vencCfg.picFormat);

        std::string enTypeStr = get_strvalue(in["venc_config"], "encode_type", "PT_H264");
        std::map<std::string, acldvppStreamFormat>::iterator streamFormat_iter = g_type2enum_map.find(enTypeStr);
        if (streamFormat_iter == g_type2enum_map.end()) {
            LOG_ERROR("[INFO]not found enType[%s] in map g_type2enum_map", enTypeStr.c_str());
        }
        cfg->vencCfg.enType = streamFormat_iter->second;
        LOG_INFO("vencCfg.enType[%d]", (int)cfg->vencCfg.enType);
#endif

        std::string rcModeStr = get_strvalue(in["venc_config"], "rcMode", "CBR");
        std::map<std::string, int>::iterator rcMode_iter = g_rcMode2enum_map.find(rcModeStr);
        if (rcMode_iter == g_rcMode2enum_map.end()) {
            LOG_ERROR("[INFO]not found enType[%s] in map g_rcMode2enum_map", rcModeStr.c_str());
        }
        cfg->vencCfg.rcMode = rcMode_iter->second;
        LOG_INFO("vencCfg.rcMode[%d]", (int)cfg->vencCfg.rcMode);

        cfg->vencCfg.picWidth = get_intvalue(in["venc_config"], "pic_width", 1920);
        cfg->vencCfg.picHeight = get_intvalue(in["venc_config"], "pic_height", 1080);

        cfg->vencCfg.keyFrameInterval = get_intvalue(in["venc_config"], "keyFrameInterval", 16);
        if (cfg->vencCfg.keyFrameInterval == 0) {
            LOG_ERROR("[INFO]keyFrameInterval[%u] must not be zero", cfg->vencCfg.keyFrameInterval);
        }

        cfg->vencCfg.srcRate = get_intvalue(in["venc_config"], "src_rate", 30);
        if (cfg->vencCfg.srcRate < VENC_SRC_RATE_MIN || cfg->vencCfg.srcRate > VENC_SRC_RATE_MAX) {
            LOG_ERROR("[INFO]srcRate[%d] must in scope[%d, %d]", cfg->vencCfg.srcRate, VENC_SRC_RATE_MIN,
                VENC_SRC_RATE_MAX);
        }

        cfg->vencCfg.maxBitRate = get_intvalue(in["venc_config"], "maxBit_rate", 300);
        if (cfg->vencCfg.maxBitRate < VENC_MAXBIT_RATE_MIN || cfg->vencCfg.maxBitRate > VENC_MAXBIT_RATE_MAX) {
            LOG_ERROR("[INFO]maxBitRate[%d] must in scope[%d, %d]", cfg->vencCfg.maxBitRate, VENC_MAXBIT_RATE_MIN,
                VENC_MAXBIT_RATE_MAX);
        }

        cfg->vencCfg.outFolder = get_strvalue(in["venc_config"], "outFolder", "./venc_result_folder");
        cfg->vencCfg.outFileNumMax = get_intvalue(in["venc_config"], "outFileNum_max", 10);
        cfg->vencCfg.memPoolSize = get_intvalue(in["venc_config"], "mem_pool_size", 64);
    }
    if (cfg->commCfg.inferFlag == 1) {
        cfg->inferCfg.batch_size = get_intvalue(in["infer_config"], "batch_size", 1);
        LOG_INFO("batch_size[%d]", cfg->inferCfg.batch_size);

        cfg->inferCfg.infer_type = get_intvalue(in["infer_config"], "infer_type", 1);
        LOG_INFO("infer_type[%d]", cfg->inferCfg.infer_type);

        cfg->inferCfg.channelNum = get_intvalue(in["infer_config"], "channel_num", 1);
        LOG_INFO("channelNum[%d]", cfg->inferCfg.channelNum);

        cfg->inferCfg.postType = get_intvalue(in["infer_config"], "post_type", 0);
        LOG_INFO("postType[%d]", cfg->inferCfg.postType);

        cfg->inferCfg.imgType = get_strvalue(in["infer_config"], "input_data_type", "yuv");
        LOG_INFO("imgType[%s]", cfg->inferCfg.imgType.c_str());

        cfg->inferCfg.modelNum = get_intvalue(in["infer_config"]["model_config"], "model_num", 1);
        LOG_INFO("modelNum[%d]", cfg->inferCfg.modelNum);

        for (int i = 0; i < cfg->inferCfg.modelNum; i++) {
            std::string omPath = get_strArrayvalue(in["infer_config"]["model_config"], "om_path_list",
                "../model/resnet/resnet50_aipp_b1_fp16_output_FP32.om", i);
            LOG_INFO("omPath[%s]", omPath.c_str());

            std::string modelType = get_strArrayvalue(in["infer_config"]["model_config"], "model_type", "resnet50", i);
            LOG_INFO("model_type[%s]", modelType.c_str());

            std::string resultFolderPath =
                get_strArrayvalue(in["infer_config"]["model_config"], "result_path_list", "../model1", i);
            LOG_INFO("resultFolderPath[%s]", resultFolderPath.c_str());

            cfg->inferCfg.omPatch.push_back(omPath);
            cfg->inferCfg.modelType.push_back(modelType);
            cfg->inferCfg.resultFolderPath.push_back(resultFolderPath);
            cfg->inferCfg.inputFilePath.push_back(cfg->dataCfg.dir_path_vec[i]);
        }

        cfg->inferCfg.resnetStdFile = get_strvalue(in["infer_config"], "resnet_std_file", "../datasets/input_1024.csv");
        LOG_INFO("resnetStdFile[%s]", cfg->inferCfg.resnetStdFile.c_str());
        cfg->inferCfg.yoloImgInfoFile =
            get_strvalue(in["infer_config"], "yolov3_img_info_file", "../datasets/configure/yolov3Config_coco_yuv");
        LOG_INFO("yoloImgInfoFile[%s]", cfg->inferCfg.yoloImgInfoFile.c_str());

        if (cfg->inferCfg.infer_type != 0) {
            cfg->inferCfg.inferAsynPara.mem_pool_size =
                get_intvalue(in["infer_config"]["Exec_asyn_para"], "mem_pool_size", 32);
            LOG_INFO("cfg->inferCfg.inferAsynPara.mem_pool_size[%d]", cfg->inferCfg.inferAsynPara.mem_pool_size);
            cfg->inferCfg.inferAsynPara.callback_interval =
                get_intvalue(in["infer_config"]["Exec_asyn_para"], "callback_interval", 4);
            LOG_INFO("cfg->inferCfg.inferAsynPara.callback_interval[%d]",
                cfg->inferCfg.inferAsynPara.callback_interval);
        }

        cfg->inferCfg.dynamicBathFlag = get_intvalue(in["infer_config"], "dynamicBatch_flag", 0);
        LOG_INFO("cfg->inferCfg.dynamicBathFlag[%d]", cfg->inferCfg.dynamicBathFlag);

        cfg->inferCfg.dynamicImgFlag = get_intvalue(in["infer_config"], "dynamicImg_flag", 0);
        LOG_INFO("cfg->dynamicImgFlag[%d]", cfg->inferCfg.dynamicImgFlag);

        if (cfg->inferCfg.dynamicImgFlag == 1) {
            cfg->inferCfg.dynamicImg.shapeW =
                get_intvalue(in["infer_config"]["dynamic_img_config"], "shape_weight", 224);
            LOG_INFO("cfg->inferCfg.dynamicImg.shapeW[%d]", cfg->inferCfg.dynamicImg.shapeW);

            cfg->inferCfg.dynamicImg.shapeH =
                get_intvalue(in["infer_config"]["dynamic_img_config"], "shape_height", 224);
            LOG_INFO("cfg->inferCfg.dynamicImg.shapeH[%d]", cfg->inferCfg.dynamicImg.shapeH);
        }

        cfg->inferCfg.dynamicAippFlag = get_intvalue(in["infer_config"], "dynamicAipp_flag", 0);
        LOG_INFO("cfg->inferCfg.dynamicAippFlag[%d]", cfg->inferCfg.dynamicAippFlag);

        if (cfg->inferCfg.dynamicAippFlag == 1) {
            std::string tmp = in["infer_config"]["dynamic_AIPP_config"]["inputFormat"];
            cfg->inferCfg.dynamicAippCfg.inputFormat = g_inputFormat_map[tmp];
            cfg->inferCfg.dynamicAippCfg.srcImageSizeW = in["infer_config"]["dynamic_AIPP_config"]["srcImageSizeW"];
            cfg->inferCfg.dynamicAippCfg.srcImageSizeH = in["infer_config"]["dynamic_AIPP_config"]["srcImageSizeH"];
            cfg->inferCfg.dynamicAippCfg.rbuvSwapSwitch = in["infer_config"]["dynamic_AIPP_config"]["rbuvSwapSwitch"];
            cfg->inferCfg.dynamicAippCfg.axSwapSwitch = in["infer_config"]["dynamic_AIPP_config"]["axSwapSwitch"];

            cfg->inferCfg.dynamicAippCfg.cscParams.csc_switch =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["csc_switch"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR0C0 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR0C0"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR0C1 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR0C1"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR0C2 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR0C2"];

            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR1C0 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR1C0"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR1C1 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR1C1"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR1C2 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR1C2"];

            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR2C0 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR2C0"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR2C1 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR2C1"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscMatrixR2C2 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscMatrixR2C2"];

            cfg->inferCfg.dynamicAippCfg.cscParams.cscOutputBiasR0 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscOutputBiasR0"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscOutputBiasR1 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscOutputBiasR1"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscOutputBiasR2 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscOutputBiasR2"];

            cfg->inferCfg.dynamicAippCfg.cscParams.cscInputBiasR0 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscInputBiasR0"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscInputBiasR1 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscInputBiasR1"];
            cfg->inferCfg.dynamicAippCfg.cscParams.cscInputBiasR2 =
                in["infer_config"]["dynamic_AIPP_config"]["Cscparams"]["cscInputBiasR2"];

            // scf
            cfg->inferCfg.dynamicAippCfg.scfCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.scfCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.scfParams = new aippScfConfig[cfg->inferCfg.dynamicAippCfg.scfCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.scfCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.scfParams[i].scfSwitch =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["scfSwitch"][i];
                cfg->inferCfg.dynamicAippCfg.scfParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["batchIndex"][i];
                cfg->inferCfg.dynamicAippCfg.scfParams[i].scfInputSizeW =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["scfInputSizeW"][i];
                cfg->inferCfg.dynamicAippCfg.scfParams[i].scfInputSizeH =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["scfInputSizeH"][i];
                cfg->inferCfg.dynamicAippCfg.scfParams[i].scfOutputSizeW =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["scfOutputSizeW"][i];
                cfg->inferCfg.dynamicAippCfg.scfParams[i].scfOutputSizeH =
                    in["infer_config"]["dynamic_AIPP_config"]["scfParams"]["scfOutputSizeH"][i];
            }
            // crop
            cfg->inferCfg.dynamicAippCfg.cropCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.cropCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.cropParams = new aippCropConfig[cfg->inferCfg.dynamicAippCfg.cropCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.cropCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.cropParams[i].cropSwitch =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["cropSwitch"][i];
                cfg->inferCfg.dynamicAippCfg.cropParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["batchIndex"][i];
                cfg->inferCfg.dynamicAippCfg.cropParams[i].cropStartPosW =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["cropStartPosW"][i];
                cfg->inferCfg.dynamicAippCfg.cropParams[i].cropStartPosH =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["cropStartPosH"][i];
                cfg->inferCfg.dynamicAippCfg.cropParams[i].cropSizeW =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["cropSizeW"][i];
                cfg->inferCfg.dynamicAippCfg.cropParams[i].cropSizeH =
                    in["infer_config"]["dynamic_AIPP_config"]["cropParams"]["cropSizeH"][i];
            }

            cfg->inferCfg.dynamicAippCfg.padCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.padCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.paddingParams =
                    new aippPaddingConfig[cfg->inferCfg.dynamicAippCfg.padCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.padCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].paddingSwitch =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["paddingSwitch"][i];
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].paddingSizeTop =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["paddingSizeTop"][i];
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].paddingSizeBottom =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["paddingSizeBottom"][i];
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].paddingSizeLeft =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["paddingSizeLeft"][i];
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].paddingSizeRight =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["paddingSizeRight"][i];
                cfg->inferCfg.dynamicAippCfg.paddingParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["paddingParams"]["batchIndex"][i];
            }

            cfg->inferCfg.dynamicAippCfg.dtcPixelMeanCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.dtcPixelMeanCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams =
                    new aippDtcPixelMeanConfig[cfg->inferCfg.dynamicAippCfg.dtcPixelMeanCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.dtcPixelMeanCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams[i].dtcPixelMeanChn0 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["dtcPixelMeanChn0"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams[i].dtcPixelMeanChn1 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["dtcPixelMeanChn1"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams[i].dtcPixelMeanChn2 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["dtcPixelMeanChn2"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams[i].dtcPixelMeanChn3 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["dtcPixelMeanChn3"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMeanParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMeanParams"]["batchIndex"][i];
            }

            cfg->inferCfg.dynamicAippCfg.dtcPixelMinCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.dtcPixelMinCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams =
                    new aippDtcPixelMinConfig[cfg->inferCfg.dynamicAippCfg.dtcPixelMinCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.dtcPixelMinCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams[i].dtcPixelMinChn0 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["dtcPixelMinChn0"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams[i].dtcPixelMinChn1 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["dtcPixelMinChn1"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams[i].dtcPixelMinChn2 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["dtcPixelMinChn2"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams[i].dtcPixelMinChn3 =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["dtcPixelMinChn3"][i];
                cfg->inferCfg.dynamicAippCfg.dtcPixelMinParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["dtcPixelMinParams"]["batchIndex"][i];
            }

            cfg->inferCfg.dynamicAippCfg.pixelVarReciCfgNum =
                in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["batchIndex"].size();
            if (cfg->inferCfg.dynamicAippCfg.pixelVarReciCfgNum > 0) {
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams =
                    new aippPixelVarReciConfig[cfg->inferCfg.dynamicAippCfg.pixelVarReciCfgNum];
            }
            for (int i = 0; i < cfg->inferCfg.dynamicAippCfg.pixelVarReciCfgNum; i++) {
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams[i].dtcPixelVarReciChn0 =
                    in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["dtcPixelVarReciChn0"][i];
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams[i].dtcPixelVarReciChn1 =
                    in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["dtcPixelVarReciChn1"][i];
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams[i].dtcPixelVarReciChn2 =
                    in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["dtcPixelVarReciChn2"][i];
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams[i].dtcPixelVarReciChn3 =
                    in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["dtcPixelVarReciChn3"][i];
                cfg->inferCfg.dynamicAippCfg.pixelVarReciParams[i].batchIndex =
                    in["infer_config"]["dynamic_AIPP_config"]["pixelVarReciParams"]["batchIndex"][i];
            }
        }

        if (in["infer_config"], "dynamicDims_flag" != nullptr) {
            cfg->inferCfg.dynamicDimsFlag = get_intvalue(in["infer_config"], "dynamicDims_flag", 0);
            LOG_INFO("cfg->inferCfg.dynamicDimsFlag[%d]", cfg->inferCfg.dynamicDimsFlag);
            if (cfg->inferCfg.dynamicDimsFlag == 1) {
                cfg->inferCfg.dynamicDims.dydims.dimCount = in["infer_config"]["dynamic_dims_config"]["dimCount"];
                for (int i = 0; i < cfg->inferCfg.dynamicDims.dydims.dimCount; i++) {
                    std::stringstream ss;
                    ss << i;
                    std::string s1 = "dim[" + ss.str() + "]";
                    cfg->inferCfg.dynamicDims.dydims.dims[i] = in["infer_config"]["dynamic_dims_config"][s1];
                }
            }
        }
    }

    cfg->commCfg.vdecFlag = get_intvalue(in["common"], "vdec_flag", 0);
    LOG_INFO("cfg->commCfg.vdecFlag[%u]", cfg->commCfg.vdecFlag);
    cfg->commCfg.jpegdFlag = get_intvalue(in["common"], "jpegd_flag", 0);
    LOG_INFO("cfg->commCfg.jpegdFlag[%u]", cfg->commCfg.jpegdFlag);
    cfg->commCfg.pngdFlag = get_intvalue(in["common"], "pngd_flag", 0);
    LOG_INFO("cfg->commCfg.pngdFlag[%u]", cfg->commCfg.pngdFlag);

    if (cfg->commCfg.vdecFlag == 1 && cfg->commCfg.jpegdFlag == 1) {
        LOG_ERROR("vdec_flag[%u] and jpegd_flag[%u] can not config at same time", cfg->commCfg.vdecFlag,
            cfg->commCfg.jpegdFlag);
        return;
    }

    if (cfg->commCfg.vdecFlag == 1 && cfg->commCfg.pngdFlag == 1) {
        LOG_ERROR("vdec_flag[%u] and pngd_flag[%u] can not config at same time", cfg->commCfg.vdecFlag,
            cfg->commCfg.pngdFlag);
        return;
    }

    if (cfg->commCfg.jpegdFlag == 1 && cfg->commCfg.pngdFlag == 1) {
        LOG_ERROR("jpegd_flag[%u] and pngd_flag[%u] can not config at same time", cfg->commCfg.jpegdFlag,
            cfg->commCfg.pngdFlag);
        return;
    }

    printf("vdecFlag:%u, jpegdFlag:%u, pngdFlag:%u\n", cfg->commCfg.vdecFlag, cfg->commCfg.jpegdFlag,
        cfg->commCfg.pngdFlag);

    if (cfg->commCfg.vdecFlag == 1 || cfg->commCfg.jpegdFlag == 1 || cfg->commCfg.pngdFlag == 1) {
        cfg->vdecCfgPara.channelNum = get_intvalue(in["vdec_jpegd_config"], "channel_num", 1);
        LOG_INFO("cfg->vdecCfgPara.channelNum[%d]", cfg->vdecCfgPara.channelNum);

        cfg->vdecCfgPara.width = get_intvalue(in["vdec_jpegd_config"], "frame_width", 3840);
        LOG_INFO("cfg->vdecCfgPara.width[%d]", cfg->vdecCfgPara.width);
        cfg->vdecCfgPara.height = get_intvalue(in["vdec_jpegd_config"], "frame_height", 2161);
        LOG_INFO("cfg->vdecCfgPara.height[%d]", cfg->vdecCfgPara.height);
        cfg->vdecCfgPara.outWidth = get_intvalue(in["vdec_jpegd_config"], "out_width", 1920);
        LOG_INFO("cfg->vdecCfgPara.outWidth[%d]", cfg->vdecCfgPara.outWidth);
        cfg->vdecCfgPara.outHeight = get_intvalue(in["vdec_jpegd_config"], "out_height", 1080);
        LOG_INFO("cfg->vdecCfgPara.outHeight[%d]", cfg->vdecCfgPara.outHeight);

        cfg->vdecCfgPara.milliSec = get_intvalue(in["vdec_jpegd_config"], "time_out_milliSec", 200);
        cfg->vdecCfgPara.intervalTime = get_intvalue(in["vdec_jpegd_config"], "send_interval_milliSec", 200);

        std::string enTypeStr = get_strvalue(in["vdec_jpegd_config"], "Type", "PT_JPEG");
        std::string pixelFormatStr =
            get_strvalue(in["vdec_jpegd_config"], "pixel_format", "PIXEL_FORMAT_YUV_SEMIPLANAR_420");

#ifdef VERSION_HOST

        if (cfg->commCfg.vdecFlag == 1) {
            std::map<std::string, acldvppStreamFormat>::iterator streamFormat_iter = g_type2enum_map.find(enTypeStr);
            if (streamFormat_iter == g_type2enum_map.end()) {
                LOG_ERROR("[INFO]not found enType[%s] in map g_type2enum_map", enTypeStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enType = streamFormat_iter->second;
        } else if (cfg->commCfg.jpegdFlag == 1) {
            std::map<std::string, JPEGD_TYPE>::iterator jpegd_iter = g_jpegdType_map.find(enTypeStr);
            if (jpegd_iter == g_jpegdType_map.end()) {
                LOG_ERROR("[INFO]not found enType[%s] in map g_jpegdType_map", enTypeStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enType = jpegd_iter->second;
        }

        LOG_INFO("vdecCfg->enType[%d]", (int)cfg->vdecCfgPara.enType);

        std::map<std::string, acldvppPixelFormat>::iterator outPicFormat_iter =
            g_pixelFormat2enum_map.find(pixelFormatStr);
        if (outPicFormat_iter == g_pixelFormat2enum_map.end()) {
            LOG_ERROR("[INFO]not found enType[%s] in map g_payloadType2enum_map", pixelFormatStr.c_str());
            return;
        }
        cfg->vdecCfgPara.enPixelFormat = outPicFormat_iter->second;
        LOG_INFO("vdecCfg->enPixelFormat[%d]", (int)cfg->vdecCfgPara.enPixelFormat);
#endif

#ifdef VERSION_CTRL_CPU_1951

        cfg->vdecCfgPara.refFrameNum = get_intvalue(in["vdec_jpegd_config"], "ref_frame_num", 7);
        cfg->vdecCfgPara.displayFrameNum = get_intvalue(in["vdec_jpegd_config"], "display_frame_num", 2);
        cfg->vdecCfgPara.apha = get_intvalue(in["vdec_jpegd_config"], "apha_value", 255);

        std::string enModeStr = get_strvalue(in["vdec_jpegd_config"], "mode", "VIDEO_MODE_FRAME");
        std::string enDecModeStr = get_strvalue(in["vdec_jpegd_config"], "decode_mode", "VIDEO_DEC_MODE_IP");
        std::string enBitWidthStr = get_strvalue(in["vdec_jpegd_config"], "bit_width_type", "DATA_BITWIDTH_8");

        std::string enCompressModeStr = get_strvalue(in["vdec_jpegd_config"], "compress_mode", "COMPRESS_MODE_NONE");
        std::string enVedioFormatStr = get_strvalue(in["vdec_jpegd_config"], "vedio_format", "VIDEO_FORMAT_TILE_64x16");
        std::string enOutputOrderStr = get_strvalue(in["vdec_jpegd_config"], "output_order", "VIDEO_OUTPUT_ORDER_DEC");

        std::map<std::string, PAYLOAD_TYPE_E>::iterator payloadType_iter = g_payloadType2enum_map.find(enTypeStr);
        if (payloadType_iter == g_payloadType2enum_map.end()) {
            LOG_ERROR("[INFO]not found enType[%s] in map g_payloadType2enum_map", enTypeStr.c_str());
            return;
        }
        cfg->vdecCfgPara.enType = payloadType_iter->second;
        LOG_INFO("vdecCfg->enType[%d]", (int)cfg->vdecCfgPara.enType);

        // output picture format
        std::map<std::string, PIXEL_FORMAT_E>::iterator outPicFormat_iter = g_pixelFormat2enum_map.find(pixelFormatStr);
        if (outPicFormat_iter == g_pixelFormat2enum_map.end()) {
            LOG_ERROR("[INFO]not found enPixelFormat[%s] in map g_pixelFormat2enum_map", pixelFormatStr.c_str());
            return;
        }
        cfg->vdecCfgPara.enPixelFormat = outPicFormat_iter->second;
        LOG_INFO("vdecCfg->enPixelFormat[%d]", (int)cfg->vdecCfgPara.enPixelFormat);

        std::map<std::string, VIDEO_MODE_E>::iterator mode_iter = g_mode2enum_map.find(enModeStr);
        if (mode_iter == g_mode2enum_map.end()) {
            LOG_ERROR("[INFO]not found enMode[%s] in map g_mode2enum_map", enModeStr.c_str());
            return;
        }
        cfg->vdecCfgPara.enMode = mode_iter->second;
        LOG_INFO("vdecCfg->enMode[%d]", (int)cfg->vdecCfgPara.enMode);

        if (jsonCfg.commCfg.vdecFlag == 1) {
            // decode mode
            std::map<std::string, VIDEO_DEC_MODE_E>::iterator dec_mode_iter = g_decMode2enum_map.find(enDecModeStr);
            if (dec_mode_iter == g_decMode2enum_map.end()) {
                LOG_ERROR("[INFO]not found enMode[%s] in map g_decMode2enum_map", enDecModeStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enDecMode = dec_mode_iter->second;
            LOG_INFO("vdecCfg->enDecMode[%d]", (int)cfg->vdecCfgPara.enDecMode);

            // bit width
            std::map<std::string, DATA_BITWIDTH_E>::iterator bitWidth_iter = g_bitWidth2enum_map.find(enBitWidthStr);
            if (bitWidth_iter == g_bitWidth2enum_map.end()) {
                LOG_ERROR("[INFO]not found enMode[%s] in map g_bitWidth2enum_map", enBitWidthStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enBitWidth = bitWidth_iter->second;
            LOG_INFO("vdecCfg->enBitWidth[%d]", (int)cfg->vdecCfgPara.enBitWidth);

            // compress mode
            std::map<std::string, COMPRESS_MODE_E>::iterator compressMode_iter =
                g_compressMode2enum_map.find(enCompressModeStr);
            if (compressMode_iter == g_compressMode2enum_map.end()) {
                LOG_ERROR("[INFO]not found enCompressMode[%s] in map g_compressMode2enum_map",
                    enCompressModeStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enCompressMode = compressMode_iter->second;
            LOG_INFO("vdecCfg->enCompressMode[%d]", (int)cfg->vdecCfgPara.enCompressMode);

            // vedio format
            std::map<std::string, VIDEO_FORMAT_E>::iterator vedioFormat_iter =
                g_vedioFormat2enum_map.find(enVedioFormatStr);
            if (vedioFormat_iter == g_vedioFormat2enum_map.end()) {
                LOG_ERROR("[INFO]not found enVedioFormat[%s] in map g_vedioFormat2enum_map", enVedioFormatStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enVedioFormat = vedioFormat_iter->second;
            LOG_INFO("vdecCfg->enVedioFormat[%d]", (int)cfg->vdecCfgPara.enVedioFormat);

            // output order
            std::map<std::string, VIDEO_OUTPUT_ORDER_E>::iterator outputOrder_iter =
                g_outputOrder2enum_map.find(enOutputOrderStr);
            if (outputOrder_iter == g_outputOrder2enum_map.end()) {
                LOG_ERROR("[INFO]not found enVedioFormat[%s] in map g_outputOrder2enum_map", enOutputOrderStr.c_str());
                return;
            }
            cfg->vdecCfgPara.enOutputOrder = outputOrder_iter->second;
            LOG_INFO("vdecCfg->enOutputOrder[%d]", (int)cfg->vdecCfgPara.enOutputOrder);
        }

#endif

#ifdef VERSION_CTRL_CPU_1910
        std::map<std::string, std::string>::iterator enType_iter = g_payloadType2Name_map.find(enTypeStr);
        if (enType_iter == g_payloadType2Name_map.end()) {
            LOG_ERROR("[INFO]not found enMode[%s] in map g_payloadType2Name_map", enTypeStr.c_str());
            return;
        }
        cfg->vdecCfgPara.streamFormat = enType_iter->second;
        LOG_INFO("vdecCfg->enPixelFormat[%d]", cfg->vdecCfgPara.enPixelFormat);
#endif
    }

    cfg->commCfg.vpcFlag = get_intvalue(in["common"], "vpc_flag", 0);
    if (cfg->commCfg.vpcFlag == 1) {
        // 读取子节点信息 VpcParam vpc analysis
        cfg->vpcCfg.vpc_channel_num = get_intvalue(in["vpc_config"], "vpc_channel_num", 1);
        LOG_INFO("vpc_channel_num[%d]", cfg->vpcCfg.vpc_channel_num);

#ifdef VERSION_CTRL_CPU_1951
        cfg->vpcCfg.s32_millisec = get_intvalue(in["vpc_config"], "s32_millisec", 20);
        LOG_INFO("vpc_image_format[%u]", cfg->vpcCfg.s32_millisec);
        cfg->vpcCfg.interpolation = get_intvalue(in["vpc_config"], "interpolation", 0);
        LOG_INFO("interpolation[%u]", cfg->vpcCfg.interpolation);

        std::string vpc_in_format = get_strvalue(in["vpc_config"], "vpc_in_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_in_img_format = g_1951devDvppInputFormat_map[vpc_in_format];
        LOG_INFO("inputFormat %u", cfg->vpcCfg.vpc_in_img_format);
        std::string vpc_out_format = get_strvalue(in["vpc_config"], "vpc_out_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_out_img_format = g_1951devDvppInputFormat_map[vpc_out_format];
        LOG_INFO("outputFormat %u", cfg->vpcCfg.vpc_out_img_format);

        // /add by x00505833
        cfg->vpcCfg.vpcFx = get_doublevalue(in["vpc_config"], "vpcFx", 0.0);
        LOG_INFO("vpcFx[%f]", cfg->vpcCfg.vpcFx);
        cfg->vpcCfg.vpcFy = get_doublevalue(in["vpc_config"], "vpcFy", 0.0);
        LOG_INFO("vpcFy[%f]", cfg->vpcCfg.vpcFy);
#endif

#ifdef VERSION_CTRL_CPU_1910
        std::string vpc_in_format = get_strvalue(in["vpc_config"], "vpc_in_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_in_img_format = g_1910devDvppInputFormat_map[vpc_in_format];
        LOG_INFO("inputFormat %u", cfg->vpcCfg.vpc_in_img_format);
        std::string vpc_out_format = get_strvalue(in["vpc_config"], "vpc_out_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_out_img_format = g_1910devDvppOutFormat_map[vpc_out_format];
        LOG_INFO("inputFormat %u", cfg->vpcCfg.vpc_out_img_format);
#endif
#ifdef VERSION_HOST
        std::string vpc_in_format = get_strvalue(in["vpc_config"], "vpc_in_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_in_img_format = g_aclDvppInputFormat_map[vpc_in_format];
        LOG_INFO("inputFormat %u", cfg->vpcCfg.vpc_in_img_format);
        std::string vpc_out_format = get_strvalue(in["vpc_config"], "vpc_out_img_format", "YUV420SP");
        cfg->vpcCfg.vpc_out_img_format = g_aclDvppInputFormat_map[vpc_out_format];
        LOG_INFO("inputFormat %u", cfg->vpcCfg.vpc_out_img_format);
#endif

        cfg->vpcCfg.vpc_out_width = get_intvalue(in["vpc_config"], "vpc_out_width", 256);
        LOG_INFO("vpc_out_width[%u]", cfg->vpcCfg.vpc_out_width);
        cfg->vpcCfg.vpc_out_height = get_intvalue(in["vpc_config"], "vpc_out_height", 256);
        LOG_INFO("vpc_out_height[%u]", cfg->vpcCfg.vpc_out_height);
        std::string tmp_vpc_type = get_strvalue(in["vpc_config"], "vpc_type", "resize");
        cfg->vpcCfg.vpc_type = g_vpcType[tmp_vpc_type];
        LOG_INFO("vpc_type[%u]", cfg->vpcCfg.vpc_type);

        cfg->vpcCfg.crop_area.left = get_intvalue(in["vpc_config"]["crop_area"], "left", 0);
        cfg->vpcCfg.crop_area.right = get_intvalue(in["vpc_config"]["crop_area"], "right", 255);
        cfg->vpcCfg.crop_area.top = get_intvalue(in["vpc_config"]["crop_area"], "top", 0);
        cfg->vpcCfg.crop_area.bottom = get_intvalue(in["vpc_config"]["crop_area"], "bottom", 255);

        cfg->vpcCfg.paste_area.left = get_intvalue(in["vpc_config"]["paste_area"], "left", 16);
        cfg->vpcCfg.paste_area.right = get_intvalue(in["vpc_config"]["paste_area"], "right", 255);
        cfg->vpcCfg.paste_area.top = get_intvalue(in["vpc_config"]["paste_area"], "top", 0);
        cfg->vpcCfg.paste_area.bottom = get_intvalue(in["vpc_config"]["paste_area"], "bottom", 255);

        cfg->vpcCfg.batch_pic_num = get_intvalue(in["vpc_config"], "batch_pic_num", 1);
        cfg->vpcCfg.roi_num = get_intvalue(in["vpc_config"], "roi_num", 1);
        cfg->vpcCfg.save_dvpp_file_flg = get_intvalue(in["vpc_config"], "save_dvpp_file_flg", 0);
        LOG_INFO("save_dvpp_file_flg[%d]", cfg->vpcCfg.save_dvpp_file_flg);
        cfg->vpcCfg.save_dvpp_file_path =
            get_strvalue(in["vpc_config"], "save_dvpp_file_path", "../datasets/YUV444SP_VPC_CROP_PASTE_OUT/");
        LOG_INFO("save_dvpp_file_path[%s]", cfg->vpcCfg.save_dvpp_file_path.c_str());

#ifdef VERSION_CTRL_CPU_1951
        if (cfg->vpcCfg.vpc_type == 9 || cfg->vpcCfg.vpc_type == 10) {
            cfg->vpcCfg.border_info.top = get_intvalue(in["vpc_config"]["border_info"], "top", 0);
            cfg->vpcCfg.border_info.bottom = get_intvalue(in["vpc_config"]["border_info"], "bottom", 0);
            cfg->vpcCfg.border_info.left = get_intvalue(in["vpc_config"]["border_info"], "left", 0);
            cfg->vpcCfg.border_info.right = get_intvalue(in["vpc_config"]["border_info"], "right", 0);

            std::string tmp = get_strvalue(in["vpc_config"]["border_info"], "borderType", "BORDER_CONSTANT");
            cfg->vpcCfg.border_info.borderType = g_1951devBordType_map[tmp];

            for (int i = 0; i < 4; i++) {
                cfg->vpcCfg.border_info.scalarValue.val[i] =
                    get_doubleArrayvalue(in["vpc_config"]["border_info"], "scalarValue", 0.0, i);
            }
        }
#endif

        if (cfg->vpcCfg.save_dvpp_file_flg == true) {
            DIR *op = opendir(cfg->vpcCfg.save_dvpp_file_path.c_str());
            if (NULL != op) {
                snprintf(cmd, sizeof(cmd), "rm -rf %s", cfg->vpcCfg.save_dvpp_file_path.c_str());
                printf("rm: %s\n", cmd);
                system(cmd);
            }
            memset(cmd, 0, sizeof(cmd));
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", cfg->vpcCfg.save_dvpp_file_path.c_str());
            printf("mkdir: %s\n", cmd);
            system(cmd);
        }
    }
}
