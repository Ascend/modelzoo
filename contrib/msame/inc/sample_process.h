/*
* @file sample_process.h
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
#ifndef _SAMPLE_PROCESS_H_
#define _SAMPLE_PROCESS_H_
#include "acl/acl.h"
#include "utils.h"
#include <stdio.h>

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
    Result InitResource();

    /**
    * @brief sample process
    * @return result
    */
    Result Process(std::map<char, std::string>& params, std::vector<std::string>& inputs);

private:
    void DestroyResource();

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
};
#endif
