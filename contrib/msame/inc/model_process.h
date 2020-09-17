/*
 * @file model_process.h
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: model_process
 * Author: fuyangchenghu
 * Create: 2020/6/22
 * Notes:
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _MODEL_PROCESS_H_
#define _MODEL_PROCESS_H_
#include "acl/acl.h"
#include "utils.h"
#include <string>

/**
 * ModelProcess
 */
class ModelProcess {
public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    ~ModelProcess();

    /**
    * @brief load model from file with mem
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModelFromFileWithMem(const std::string& modelPath);

    /**
    * @brief unload model
    */
    void Unload();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateDesc();

    /**
    * @brief PrintDesc
    */
    Result PrintDesc();

    /**
    * @brief destroy desc
    */
    void DestroyDesc();

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result CreateInput(void* inputDataBuffer, size_t bufferSize);

    /**
    * @brief create model input
    * @return result
    */
    Result CreateZeroInput();

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief dump model output result to file
    */
    void DumpModelOutputResult();

    /**
    * @brief get model output result
    */
    void OutputModelResult(std::string& s, std::string& modelName, size_t index);

private:
    uint32_t modelId_;
    size_t modelMemSize_;
    size_t modelWeightSize_;
    void* modelMemPtr_;
    void* modelWeightPtr_;
    bool loadFlag_; // model load flag
    aclmdlDesc* modelDesc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;
    size_t numInputs_;
    size_t numOutputs_;
};
#endif