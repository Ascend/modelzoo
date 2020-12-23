/**
* @file utils.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>
#include "common.h"
#include <map>
#include "acl/acl_base.h"
//#include "Config.h"
/**
* Utils
*/

struct aclmdlTensorDesc {
    aclmdlTensorDesc() : name(""), size(0), format(ACL_FORMAT_UNDEFINED), dataType(ACL_DT_UNDEFINED) {}
    std::string name;
    size_t size;
    aclFormat format;
    aclDataType dataType;
    std::vector<int64_t> dims;
};

struct aclmdlDesc {
    void Clear()
    {
        inputDesc.clear();
        outputDesc.clear();
        dynamicBatch.clear();
        dynamicHW.clear();
    }
    std::vector<aclmdlTensorDesc> inputDesc;
    std::vector<aclmdlTensorDesc> outputDesc;
    std::vector<uint64_t> dynamicBatch;
    std::vector<std::vector<uint64_t>> dynamicHW;
};



class Utils {
public:
    /**
    * @brief create device buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return device buffer of file
    */
    static void *GetDeviceBufferOfFile(std::string fileName, uint32_t &fileSize);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return buffer of pic
    */
    static void* ReadBinFile(std::string fileName, uint32_t& fileSize);

    static void* GetDeviceBufferOfptr(void* fileName, uint32_t fileSize);

    static long getCurrentTime();

};

#pragma once
