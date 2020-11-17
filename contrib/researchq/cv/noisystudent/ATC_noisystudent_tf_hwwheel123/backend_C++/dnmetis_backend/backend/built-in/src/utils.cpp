/**
* @file utils.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include "acl/acl.h"
#include <stdio.h>
#include <sys/time.h>

bool g_isDevice = false;

std::map<aclDataType, std::string> ACLdt{{ ACL_DT_UNDEFINED , "undefined"},
{ ACL_FLOAT , "float"},
{ ACL_FLOAT16 , "float16"},
{ ACL_INT8 , "int8"},
{ ACL_INT32 , "int32"},
{ ACL_UINT8 , "uint8"},
{ ACL_INT16 , "int16"},
{ ACL_UINT16 , "uint16"},
{ ACL_UINT32 , "uint32"},
{ ACL_INT64 , "int64"},
{ ACL_UINT64 , "uint64"},
{ ACL_DOUBLE , "double"},
{ ACL_BOOL , "bool"}};
std::map<aclDataType, int> ACLdt_size{{ ACL_DT_UNDEFINED , -1},
{ ACL_FLOAT , 4},
{ ACL_FLOAT16 , 2},
{ ACL_INT8 , 1},
{ ACL_INT32 , 4},
{ ACL_UINT8 , 1},
{ ACL_INT16 , 2},
{ ACL_UINT16 , 2},
{ ACL_UINT32 , 4},
{ ACL_INT64 , 8},
{ ACL_UINT64 , 8},
{ ACL_DOUBLE , 8},
{ ACL_BOOL , 1}};

//Config configSettings("cfg/built-in_config.txt");

void* Utils::ReadBinFile(std::string fileName, uint32_t &fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    //ERROR_LOG("binFileBufferLen is %d", binFileBufferLen);

    binFile.seekg(0, binFile.beg);

    void* binFileBufferData = nullptr;
    aclError ret = ACL_ERROR_NONE;
    if (!g_isDevice) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (binFileBufferData == nullptr) {
            ERROR_LOG("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    }
    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

void* Utils::GetDeviceBufferOfFile(std::string fileName, uint32_t &fileSize)
{
    uint32_t inputHostBuffSize = 0;
    void* inputHostBuff = Utils::ReadBinFile(fileName, inputHostBuffSize);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }
    if (!g_isDevice) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        aclError ret = aclrtMalloc(&inBufferDev, inBufferSize, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                inBufferSize, inputHostBuffSize);
            aclrtFree(inBufferDev);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }
        aclrtFreeHost(inputHostBuff);
        fileSize = inBufferSize;
        return inBufferDev;
    } else {
        fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}

void* Utils::GetDeviceBufferOfptr(void* fileName, uint32_t len)
{
    uint32_t inputHostBuffSize = len;
    void* inputHostBuff = fileName;
    if (inputHostBuff == nullptr) {
        return nullptr;
    }
    if (!g_isDevice) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        //INFO_LOG("start memcpy::aclrtMalloc is %d", Utils::getCurrentTime());
        aclError ret = aclrtMalloc(&inBufferDev, inBufferSize, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
        //INFO_LOG("end  memcpy::aclrtMalloc is %d", Utils::getCurrentTime());
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }
        //INFO_LOG("start memcpy::aclrtMemcpy is %d", Utils::getCurrentTime());
        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        //INFO_LOG("end memcpy::aclrtMemcpy is %d", Utils::getCurrentTime());
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                inBufferSize, inputHostBuffSize);
            aclrtFree(inBufferDev);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }
        //INFO_LOG("start memcpy::aclrtFreeHost is %d", Utils::getCurrentTime());
        aclrtFreeHost(inputHostBuff);
        //INFO_LOG("end memcpy::aclrtFreeHost is %d", Utils::getCurrentTime());
        //fileSize = inBufferSize;
        return inBufferDev;
    } else {
        //fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}


long Utils::getCurrentTime()
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
  // return tv.tv_usec;
}
