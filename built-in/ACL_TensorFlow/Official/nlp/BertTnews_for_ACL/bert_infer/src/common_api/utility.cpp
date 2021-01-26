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

#include "utility.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

bool g_isDevice = false;
/*
 * @brief Obtain the file name and number of files
 * @param [in] inputDirectory: Entering the folder path
 * @param [out] fileList: Names of all files in the path
 * @return Number of files
 */
int SdkInferScanFiles(std::vector<std::string> &fileList, std::string inputDirectory)
{
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char *str = inputDirectory.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL) {
        LOG_ERROR("Open directory[%s] failed.", str);
        return -1;
    }

    struct dirent *p_dirent;
    while (p_dirent = readdir(p_dir)) {
        std::string tmpFileName = p_dirent->d_name;

        if (tmpFileName == "." || tmpFileName == "..") {
            continue;
        } else if (p_dirent->d_type == 8) {
            fileList.push_back(tmpFileName);
        } else if (p_dirent->d_type == 10) {
            continue;
        } else if (p_dirent->d_type == 4) {
            continue;
        } else {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);

    if (fileList.size() == 0) {
        LOG_ERROR("No file in the directory[%s]", str);
    }

    return fileList.size();
}

/*
 * @brief Reading the bin file
 * @param [in] fileName: file name
 * @param [out] fileSize: file's size
 * @return file's context
 */
char *SdkInferReadBinFile(std::string fileName, uint32_t &fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);
    char *binFileBufferData = new (std::nothrow) char[binFileBufferLen];
    if (binFileBufferData == nullptr) {
        binFile.close();
        delete binFileBufferData;
        return nullptr;
    }

    binFile.read(binFileBufferData, binFileBufferLen);
    fileSize = binFileBufferLen;
    binFile.close();

    return binFileBufferData;
}

aclError SdkInferWriteToFile(FILE *fileFp, void *dataDev, uint32_t dataSize, bool isDevice)
{
    if (fileFp == nullptr) {
        LOG_ERROR("fileFp is nullptr!");
        return 1;
    }

    if (dataDev == nullptr) {
        LOG_ERROR("dataDev is nullptr!");
        return 1;
    }

    // copy output to host memory
    void *data = nullptr;
    aclError ret = ACL_ERROR_NONE;
    if (!isDevice) {
        ret = aclrtMallocHost(&data, dataSize);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("malloc host data buffer failed. dataSize = %u, errorCode = %d.", dataSize,
                static_cast<int32_t>(ret));
            return 1;
        }

        ret = aclrtMemcpy(data, dataSize, dataDev, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("acl memcpy data to host failed, dataSize = %u, errorCode = %d.", dataSize,
                static_cast<int32_t>(ret));
            (void)aclrtFreeHost(data);
            return ret;
        }
    } else {
        data = dataDev;
    }

    size_t writeRet = fwrite(data, 1, dataSize, fileFp);
    if (writeRet != dataSize) {
        LOG_ERROR("need write %u bytes, but only write %zu bytes, error=%s.\n", dataSize, writeRet, strerror(errno));
        ret = 1;
    }

    if (!isDevice) {
        (void)aclrtFreeHost(data);
    }

    fflush(fileFp);

    return ret;
}

bool RunStatus::isDevice_ = false;

void *SdkInferUtils::ReadBinFile2(std::string fileName, uint32_t &fileSize)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        LOG_ERROR("failed to get file");
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG_ERROR("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        LOG_ERROR("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);

    void *binFileBufferData = nullptr;
    aclError ret = ACL_ERROR_NONE;
    if (!(RunStatus::GetDeviceStatus())) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (binFileBufferData == nullptr) {
            LOG_ERROR("malloc binFileBufferData failed. binFileBufferLen is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    } else {
        ret = acldvppMalloc(&binFileBufferData, binFileBufferLen);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("malloc device buffer failed. size is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    }
    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

void *SdkInferUtils::ReadBinFile1(std::string fileName, uint32_t &fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        LOG_ERROR("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);
    void *binFileBufferData = nullptr;
    if (!(RunStatus::GetDeviceStatus())) {
        binFileBufferData = malloc(binFileBufferLen);
        if (binFileBufferData == nullptr) {
            LOG_ERROR("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        aclError aclRet = acldvppMalloc(&binFileBufferData, binFileBufferLen);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("malloc device data buffer failed, aclRet is %d", aclRet);
            return nullptr;
        }
    }

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

void *SdkInferUtils::GetDeviceBufferOfPicture(const PicDesc &picDesc, uint32_t &devPicBufferSize)
{
    if (picDesc.picName.empty()) {
        LOG_ERROR("picture file name is empty");
        return nullptr;
    }

    uint32_t inputBuffSize = 0;
    void *inputBuff = ReadBinFile2(picDesc.picName, inputBuffSize);
    if (inputBuff == nullptr) {
        return nullptr;
    }

    if (!(RunStatus::GetDeviceStatus())) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputBuffSize;
        aclError ret = acldvppMalloc(&inBufferDev, inBufferSize);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputBuff);
            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputBuff, inputBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("memcpy failed. device buffer size is %u, input host buffer size is %u", inBufferSize,
                inputBuffSize);
            acldvppFree(inBufferDev);
            aclrtFreeHost(inputBuff);
            return nullptr;
        }
        aclrtFreeHost(inputBuff);
        devPicBufferSize = inBufferSize;
        return inBufferDev;
    } else {
        devPicBufferSize = inputBuffSize;
        LOG_INFO("memcpy. device buffer size is %u", devPicBufferSize);
        return inputBuff;
    }
}

char *SdkInferUtils::GetPicDevBuffer4JpegE(const PicDesc &picDesc, uint32_t &PicBufferSize)
{
    if (picDesc.picName.empty()) {
        LOG_ERROR("picture file name is empty");
        return nullptr;
    }

    FILE *fp = fopen(picDesc.picName.c_str(), "rb");
    if (fp == nullptr) {
        LOG_ERROR("open file %s failed", picDesc.picName.c_str());
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    uint32_t fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fileLen < PicBufferSize) {
        LOG_ERROR("need read %u bytes but file %s only %u bytes", PicBufferSize, picDesc.picName.c_str(), fileLen);
        fclose(fp);
        return nullptr;
    }

    void *inputDevBuff = nullptr;
    aclError aclRet = acldvppMalloc(&inputDevBuff, PicBufferSize);
    if (aclRet != ACL_ERROR_NONE) {
        LOG_ERROR("malloc device data buffer failed, aclRet is %d", aclRet);
        fclose(fp);
        return nullptr;
    }

    void *inputBuff = nullptr;
    size_t readSize;
    if (!(RunStatus::GetDeviceStatus())) {
        aclRet = aclrtMallocHost(&inputBuff, PicBufferSize);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("malloc host data buffer failed, aclRet is %d", aclRet);
            fclose(fp);
            (void)acldvppFree(inputDevBuff);
            return nullptr;
        }

        readSize = fread(inputBuff, sizeof(char), PicBufferSize, fp);
        if (readSize < PicBufferSize) {
            LOG_ERROR("need read file %s %u bytes, but only %zu readed", picDesc.picName.c_str(), PicBufferSize,
                readSize);
            (void)aclrtFreeHost(inputBuff);
            (void)acldvppFree(inputDevBuff);
            fclose(fp);
            return nullptr;
        }
        aclRet = aclrtMemcpy(inputDevBuff, PicBufferSize, inputBuff, PicBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("memcpy from host to device failed, aclRet is %d", aclRet);
            (void)acldvppFree(inputDevBuff);
            (void)aclrtFreeHost(inputBuff);
            fclose(fp);
            return nullptr;
        }
    } else {
        readSize = fread(inputDevBuff, sizeof(char), PicBufferSize, fp);
        if (readSize < PicBufferSize) {
            LOG_ERROR("need read file %s %u bytes, but only %zu readed", picDesc.picName.c_str(), PicBufferSize,
                readSize);
            (void)acldvppFree(inputDevBuff);
            fclose(fp);
            return nullptr;
        }
    }

    fclose(fp);
    return reinterpret_cast<char *>(inputDevBuff);
}

Result SdkInferUtils::SaveDvppOutputData(const char *fileName, void *devPtr, uint32_t dataSize)
{
    void *dataPtr = nullptr;
    aclError aclRet;
    if (!(RunStatus::GetDeviceStatus())) {
        aclRet = aclrtMallocHost(&dataPtr, dataSize);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("malloc host data buffer failed, aclRet is %d", aclRet);
            return FAILED;
        }

        aclRet = aclrtMemcpy(dataPtr, dataSize, devPtr, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("dvpp output memcpy to host failed, aclRet is %d", aclRet);
            (void)aclrtFreeHost(dataPtr);
            return FAILED;
        }
    } else {
        dataPtr = devPtr;
    }

    FILE *outFileFp = fopen(fileName, "wb+");
    if (outFileFp == nullptr) {
        LOG_ERROR("fopen out file %s failed.", fileName);
        fclose(outFileFp);
        if (!(RunStatus::GetDeviceStatus())) {
            (void)aclrtFreeHost(dataPtr);
        }
        return FAILED;
    }

    size_t writeSize = fwrite(dataPtr, sizeof(char), dataSize, outFileFp);
    if (writeSize != dataSize) {
        LOG_ERROR("need write %u bytes to %s, but only write %zu bytes.", dataSize, fileName, writeSize);
        fclose(outFileFp);
        if (!(RunStatus::GetDeviceStatus())) {
            (void)aclrtFreeHost(dataPtr);
        }
        return FAILED;
    }

    if (!(RunStatus::GetDeviceStatus())) {
        (void)aclrtFreeHost(dataPtr);
    }
    fflush(outFileFp);
    fclose(outFileFp);
    return SUCCESS;
}

char *SdkInferUtils::GetPicDevBuffer4JpegD(const PicDesc &picDesc, uint32_t &devPicBufferSize)
{
    if (picDesc.picName.empty()) {
        LOG_ERROR("picture file name is empty");
        return nullptr;
    }

    uint32_t inputBuffSize = 0;
    void *inputBuff = ReadBinFile1(picDesc.picName, inputBuffSize);
    if (inputBuff == nullptr) {
        LOG_ERROR("malloc inputHostBuff failed");
        return nullptr;
    }

    void *inBufferDev = nullptr;
    uint32_t inBufferSize = inputBuffSize;
    aclError aclRet;
    if (!(RunStatus::GetDeviceStatus())) {
        aclRet = acldvppMalloc(&inBufferDev, inBufferSize);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("malloc inBufferSize failed, aclRet is %d", aclRet);
            free(inputBuff);
            return nullptr;
        }

        aclRet = aclrtMemcpy(inBufferDev, inBufferSize, inputBuff, inputBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("memcpy from host to device failed. aclRet is %d", aclRet);
            acldvppFree(inBufferDev);
            free(inputBuff);
            return nullptr;
        }
        free(inputBuff);
    } else {
        inBufferDev = inputBuff;
    }

    devPicBufferSize = inBufferSize;
    return reinterpret_cast<char *>(inBufferDev);
}

void *SdkInferUtils::ReadBinFile(std::string fileName, uint32_t &fileSize)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        LOG_ERROR("failed to get file");
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG_ERROR("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        LOG_ERROR("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);

    void *binFileBufferData = nullptr;
    aclError ret = ACL_ERROR_NONE;
    if (!g_isDevice) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (binFileBufferData == nullptr) {
            LOG_ERROR("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("malloc device buffer failed. size is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    }
    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

void *SdkInferUtils::GetDeviceBufferOfFile(std::string fileName, uint32_t &fileSize)
{
    uint32_t inputHostBuffSize = 0;
    void *inputHostBuff = SdkInferUtils::ReadBinFile(fileName, inputHostBuffSize);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }
    if (!g_isDevice) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        aclError ret = aclrtMalloc(&inBufferDev, inBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("memcpy failed. device buffer size is %u, input host buffer size is %u", inBufferSize,
                inputHostBuffSize);
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
