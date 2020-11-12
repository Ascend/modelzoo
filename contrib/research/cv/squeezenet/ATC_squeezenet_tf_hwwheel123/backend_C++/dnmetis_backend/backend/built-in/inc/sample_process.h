/**
* @file sample_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "model_process.h"
#include<vector>
using namespace std;

/**
* SampleProcess
*/
class SampleProcess {
public:
    /**
    * @brief Constructor
    */
    SampleProcess();

    /**
    * @brief Destructor
    */
    ~SampleProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource(char* omModelPath);

    /**
    * @brief sample process
    * @return result
    */
    //Result Process(char* binfile);

    Result Process(void* binfile,int len);

    Result Process(void* binfile,int len, vector<Output_buf> &output, long &npuTime);


    Result Unload();

    //ModelProcess GetModelProcess();
    ModelProcess GetModelProcess(){return processModel;};

public:
    void DestroyResource();

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    ModelProcess processModel;
};

