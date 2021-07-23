/* 
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2099. All Rights Reserved.
 * Description: 定义数据类型
 * Author: Atlas
 * Create: 2020-02-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H
#include <map>
#include <vector>
#include <memory>
#include "block_queue.h"
#include "acl/acl.h"

struct PerfInfo {
    float throughputRate;
    float moduleLantency;
    float inferLantency; // specific for inference module
    uint32_t count;
};

enum ModelType {
    MT_VISION,
    MT_NLP,
    MT_FASTERRCNN,
    MT_NMT,
    MT_WIDEDEEP,
    MT_INVALID
};

enum ImageFormatType {
    IFT_JPG,
    IFT_JPEG,
    IFT_INVALID
};

struct DataBuf {
    std::shared_ptr<uint8_t> buf;
    uint32_t len;
};

struct ImageRawData {
    uint32_t width;
    uint32_t height;
    DataBuf data;
};

struct TextRawData {
    // infer: for mutil tensor input, not batch!
    std::vector<DataBuf> textRawData;
};

struct RawData {
    uint64_t dataId;
    bool finish;
    ModelType modelType;
    ImageRawData img;
    TextRawData text;
};

struct ModelInputData {
    uint64_t dataId;
    bool finish;
    ModelType modelType;
    ImageRawData img;
    TextRawData text;
};

// output data of batchsize
struct ModelOutputData {
    bool finish;
    std::vector<uint64_t> vDataId;
    uint32_t realNum; // the real number of batch
    std::map<std::string, DataBuf> modelOutputData;
};

extern aclrtContext context;

#endif
