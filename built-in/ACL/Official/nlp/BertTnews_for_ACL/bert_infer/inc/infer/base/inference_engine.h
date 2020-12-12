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

#ifndef _INFERENCE_ENGINE_H
#define _INFERENCE_ENGINE_H


#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"

#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cstring>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <dirent.h>
#include <stdarg.h>
#include <libgen.h>
#include <string>
#include <getopt.h>
#include <map>
#include <cerrno>
#include <errno.h>

#include "utility.h"
#include "json.h"
#include "block_queue.h"
#include "common.h"

#define MODEL_INPUT_OUTPUT_NUM_MAX (8)

typedef enum {
    infer_type_syn = 0,
    infer_type_asyn_normal,
    infer_type_asyn_dev_switch,
    infer_type_asyn_stream_wait,
    infer_type_count
} INFERENCE_TYPE;

typedef enum {
    post_type_save_file = 0,
    post_type_calc_accuracy,
    post_type_add_det_rect,
    post_type_count
} POST_PROC_TYPE;

struct ImgInfo {
    std::string imgName;
    uint32_t resizedWidth;
    uint32_t resizedHeight;
    uint32_t width;
    uint32_t height;
    ImgInfo() : resizedWidth(0), resizedHeight(0), width(0), height(0), imgName("") {};
};

typedef struct InferenceMem {
    aclmdlDataset *input;
    aclmdlDataset *output;
    aclmdlAIPP *dyAippSet;
    std::vector<std::string> *inferFile_vec;
} InferenceMem;

struct ModelIODynamicInfo {
    size_t hwCount;
    uint64_t hw[ACL_MAX_HW_NUM][2];
    size_t batchCount;
    uint64_t batch[ACL_MAX_BATCH_NUM];
};


struct ModelIOInfo {
    aclFormat Format;
    const char *Name;
    size_t size;
    size_t dimCount;
    int64_t dims[ACL_MAX_DIM_CNT];
    aclDataType Type;
};

struct YoloImgInfo {
    uint32_t resizedWidth;
    uint32_t resizedHeight;
    std::unordered_map<std::string, std::pair<float, float>> imgSizes_map;
};

struct ResnetResult {
    int top1;
    int top5;
    int total;
    std::unordered_map<std::string, int> cmp;
    ResnetResult() : top1(0), top5(0), total(0) {};
};

struct Config {
    std::string inputFolder;
    std::string resultFolder;
    std::vector<std::string> inputArray;
    uint32_t loopNum;
    uint32_t batchSize;
    INFERENCE_TYPE inferType;
    uint32_t preChnNum;
    uint32_t curChnNum;
    std::string om;
    std::string modelType;
    std::string imgType;
    std::string frameWork;
    POST_PROC_TYPE postType;
    aclrtContext context;
    uint32_t useDvpp;
    bool isDynamicBatch;
    bool isDynamicImg;
    bool isDynamicAipp;
    // xwx5322041
    bool isDynamicDims;
    std::string resnetStdFile;
    std::string yoloImgInfoFile;

    infer_asyn_parameter asynInferCfg;
    // xwx5322041
    dynamicDimsConfig dynamicDimsCfg;
    dynamicImgConfig dynamicImgCfg;
    dynamic_aipp_config dynamicAippCfg;
};


class InferEngine {
public:
    Config *cfg_;
    void *dev_ptr_;
    void *weight_ptr_;
    char *modelData_;
    uint32_t modelId_;
    aclmdlDesc *modelDesc_;
    aclrtRunMode runMode_;
    ResnetResult resnetTopRes_;
    YoloImgInfo yoloImgInfo_;

    ModelIOInfo mdInputInfo_[MODEL_INPUT_OUTPUT_NUM_MAX];
    ModelIOInfo mdOutputInfo_[MODEL_INPUT_OUTPUT_NUM_MAX];
    ModelIODynamicInfo dyModelInfo_;
    uint32_t maxBatch_;
    size_t mdInputNum_;
    size_t mdOutputNum_;
    std::string resnetSubfix_;
    Time_Cost timeCost_;
    aclmdlAIPP *aippDynamicSet_;
    bool isTransToNextThread_;

    std::vector<std::string> files_;
    std::vector<std::thread> threads_;
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inputDatas_;
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outputDatas_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;

public:
    InferEngine();
    InferEngine(Config *config);
    InferEngine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue);
    ~InferEngine();

    virtual aclError Init(Config *cfg);

    virtual aclError LoadModel();

    virtual void UnloadModel();

    virtual aclError InitImgStdValue(std::string stdFilePath);

    virtual aclError InitYolov3ImgInfo(std::string &yoloImgInfoFile);

    virtual aclError ExecInference();

    virtual aclError CreateInferInput(std::vector<std::string> &inferFile_vec);

    virtual aclError CreateInferOutput();

    virtual aclError CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec);

    virtual void DestroyDataset(aclmdlDataset *dataset);

    virtual aclError InferenceThreadProc();

    virtual int SaveInferResult(aclmdlDataset *output, std::vector<std::string> *inferFile_vec);

    virtual aclError GetModelInputOutputInfo();

    virtual aclError SetDynamicBatch();

    virtual aclError SetDynamicImg();

    virtual aclError SetDynamicAipp();

    virtual void CalcTop(std::ofstream &file_stream);

    virtual void join();

    virtual void DumpTimeCost(std::ofstream &fstream);

private:
};


class Asyn_InferEngine : public InferEngine {
public:
    aclrtStream inferStream_;
    uint32_t memCurrent_;
    uint32_t memPre_;
    bool runFlag_;
    std::vector<InferenceMem> inferMemPool_vec_;
    pthread_t callbackPid_;

public:
    Asyn_InferEngine();
    Asyn_InferEngine(Config *config);
    Asyn_InferEngine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue);
    ~Asyn_InferEngine();

    virtual aclError InitInferenceMemPool();

    virtual void DestroyInferenceMemPool();

    virtual aclError Init(Config *cfg);

    virtual aclError ExecInference();

    virtual aclError CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec);

    virtual aclError CreateInferInput(std::vector<std::string> &inferFile_vec);

    virtual aclError InferenceThreadProc();

    virtual void DumpTimeCost(std::ofstream &fstream);

    virtual aclError SetDynamicBatch();

    virtual aclError SetDynamicImg();

    virtual aclError SetDynamicAipp();

private:
};

typedef struct ModelExe_callback_data {
    uint32_t start_index;
    uint32_t end_index;
    void *inferEng;
} ModelExe_callback_data;


aclError GetInferEngineConfig(Config *cfg, uint32_t chnIndex, std::string &modelType, std::string &inputFolder,
    std::string &outputFolder, std::string &om, aclrtContext &context, inferenceJsonConfig &jsonCfg);

int GetDynamicAippParaByBatch(size_t batchIndex, dynamic_aipp_config &dyAippCfg, std::string cfgItem);


#endif
