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
#include "common.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <dirent.h>
#include "utility.h"


char *SdkInferReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        LOG_ERROR("failed to get file");
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG_ERROR("%s is not a file, please enter a file", filePath.c_str());
        return nullptr;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Open file failed. path = %s", filePath.c_str());
        return nullptr;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        LOG_ERROR("file size is 0");
        return nullptr;
    }
    if (size != bufferSize) {
        LOG_ERROR("file size is large than buffer size");
        return nullptr;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    return static_cast<char *>(buffer);
}

bool SdkInferWriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        LOG_ERROR("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_ERROR("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        LOG_ERROR("Write file Failed.");
        return false;
    }

    return true;
}

/*
 * @brief set deviceid and create context
 * @param [in] none
 * @param [out] device_vec: device's vector
 * @param [out] contex_vec: context's vector
 * @return ret 0:success !0: failed
 */
aclError SdkInferDeviceContexInit(std::vector<uint32_t> &device_vec, std::vector<aclrtContext> &contex_vec)
{
    aclError ret = ACL_ERROR_NONE;
    LOG_INFO("Enter SdkInferDeviceContexInit.");
    for (int devIndex = 0; devIndex < device_vec.size() && devIndex < DEVICE_ID_MAX; devIndex++) {
        ret = aclrtSetDevice(device_vec[devIndex]);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtSetDevice[%d] failed, ret %d.", device_vec[devIndex], ret);
            return ret;
        }

        aclrtContext context;
        ret = aclrtCreateContext(&context, device_vec[devIndex]);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]aclrtCreateContext[%d] failed, ret %d.", device_vec[devIndex], ret);
            return ret;
        }

        contex_vec.push_back(context);
    }

    return ret;
}

/*
 * @brief reset device and destory context
 * @param [in] device_vec: device's vector
 * @param [in] contex_vec: context's vector
 * @param [out] none
 * @return ret 0:success !0: failed
 */
aclError SdkInferDestoryRsc(std::vector<uint32_t> device_vec, std::vector<aclrtContext> contex_vec)
{
    aclError ret = 0;

    for (int i = 0; i < contex_vec.size(); i++) {
        ret = aclrtDestroyContext(contex_vec[i]);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("destory context failed, ret[%d].", ret);
        }
    }

    for (int i = 0; i < device_vec.size(); i++) {
        ret = aclrtResetDevice(device_vec[i]);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("reset device failed, ret[%d].", ret);
        }
    }

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclFinalize failed, ret[%d].", ret);
    }

    return ACL_ERROR_NONE;
}

/*
 * @brief destroy databuf,free data and destroy dataset
 * @param [in] dataset: model input or output dataset
 * @param [in] flag: dvppFree flag 0: aclrtFree,1:acldvppFree
 * @param [out] none
 * @return ret 0:success !0: failed
 */
aclError SdkInferDestroyDatasetResource(aclmdlDataset *dataset, uint32_t flag)
{
    aclError ret = ACL_ERROR_NONE;

    if (dataset == nullptr) {
        LOG_ERROR("dataset == null");
        return 1;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        if (dataBuffer == nullptr) {
            LOG_ERROR("dataBuffer == null");
            continue;
        }

        void *data = aclGetDataBufferAddr(dataBuffer);
        if (data != nullptr) {
            if (flag == 1) {
                if (i > 0) {
                    ret = aclrtFree(data);
                    if (ret != ACL_ERROR_NONE) {
                        LOG_ERROR("aclrtFree data failed, ret %d", ret);
                    }
                } else {
                    return 0;
                }
            } else {
                ret = aclrtFree(data);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("aclrtFree data failed, ret %d", ret);
                }
            }
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Destroy dataBuffer failed, ret %d", ret);
        }
    }

    ret = aclmdlDestroyDataset(dataset);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtFree dataset failed, ret %d", ret);
    }

    return ret;
}

/*
 * @brief load om model from memory and alloc running mem,weight mem by user
 * @param [in] modelFile: model file path and name
 * @param [out] p_modelId: model id assigned by system
 * @param [out] ppModelDesc: model desc
 * @param [out] ppModelData: memory store for model
 * @param [out] ppMem: memory addr for model running
 * @param [out] ppWeight: memory addr for model weight
 * @return ret 0:success !0: failed
 */
aclError SdkInferLoadModelFromMem(std::string &modelFile, uint32_t *p_modelId, aclmdlDesc **ppModelDesc,
    char **ppModelData, void **ppMem, void **ppWeight)
{
    size_t memSize;
    size_t weightsize;
    uint32_t modelSize = 0;
    aclError ret;

    *ppModelData = SdkInferReadBinFile(modelFile, modelSize);
    if (*ppModelData == nullptr) {
        LOG_ERROR("can't read model file");
    }

    ret = aclmdlQuerySizeFromMem(*ppModelData, modelSize, &memSize, &weightsize);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("query memory size failed, ret %d", ret);
        delete[] * ppModelData;
        return ret;
    }

    ret = aclrtMalloc(ppMem, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("aclrtMalloc run mem failed, ret %d", ret);
        delete[] * ppModelData;
        return ret;
    }

    ret = aclrtMalloc(ppWeight, weightsize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("aclrtMalloc weight memory failed, ret %d", ret);
        delete[] * ppModelData;
        aclrtFree(*ppMem);
        return ret;
    }

    ret = aclmdlLoadFromMemWithMem(*ppModelData, modelSize, p_modelId, *ppMem, memSize, *ppWeight, weightsize);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("load model from memory failed, ret %d", ret);
        delete[] * ppModelData;
        aclrtFree(*ppMem);
        aclrtFree(*ppWeight);
        return ret;
    }

    LOG_INFO("Load model success. memSize: %lu, weightSize: %lu.", memSize, weightsize);

    *ppModelDesc = aclmdlCreateDesc();
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("create model desc failed, ret %d", ret);
        delete[] * ppModelData;
        aclrtFree(*ppMem);
        aclrtFree(*ppWeight);
        aclmdlUnload(*p_modelId);
        return ret;
    }

    ret = aclmdlGetDesc(*ppModelDesc, *p_modelId);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("get model desc failed, ret %d", ret);
        delete[] * ppModelData;
        aclrtFree(*ppMem);
        aclrtFree(*ppWeight);
        aclmdlUnload(*p_modelId);
        aclmdlDestroyDesc(*ppModelDesc);
        return ret;
    }

    return ACL_ERROR_NONE;
}

/*
 * @brief unload om model and free running mem,weight mem,model mem
 * @param [in] modelFile: model file path and name
 * @param [out] modelId: model id assigned by system
 * @param [out] pModelDesc: model desc
 * @param [out] pModelData: memory store for model
 * @param [out] pMem: memory addr for model running
 * @param [out] pWeight: memory addr for model weight
 * @return ret 0:success !0: failed
 */
aclError SdkInferUnloadModelAndDestroyResource(uint32_t modelId, aclmdlDesc *pModelDesc, char *pModelData, void *pMem,
    void *pWeight)
{
    aclError ret;
    uint8_t error_flag = 0;

    if (pModelDesc != nullptr) {
        ret = aclmdlDestroyDesc(pModelDesc);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlDestroyDesc failed, ret[%d]", ret);
            error_flag += 1;
        }
    }

    if (modelId != 0) {
        ret = aclmdlUnload(modelId);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlUnload failed, ret[%d]", ret);
            error_flag += 1;
        }
    }

    if (pModelData != nullptr) {
        delete[] pModelData;
    }

    if (pMem != nullptr) {
        ret = aclrtFree(pMem);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtFree run mem failed, ret[%d]", ret);
            error_flag += 1;
        }
    }

    if (pWeight != nullptr) {
        ret = aclrtFree(pWeight);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtFree weight mem failed, ret[%d]", ret);
            error_flag += 1;
        }
    }

    if (error_flag > 0) {
        return 1;
    } else {
        return ACL_ERROR_NONE;
    }
}

long long SdkInferElapsedus(void)
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time); 
    return (long long)(time.tv_sec * 1e6 + time.tv_nsec * 1.0 / 1000);
}

bool SdkInferFolderExists(std::string foldname)
{
    DIR *dir;

    if ((dir = opendir(foldname.c_str())) == NULL) {
        return false;
    }

    closedir(dir);

    return true;
}

bool SdkInferFileExists(std::string filename)
{
    std::fstream file;
    file.open(filename, std::ios::in);

    if (!file) {
        return false;
    }

    file.close();
    return true;
}

void SdkInferGetImgWHFromJpegBuf(unsigned char *mjpeg, uint32_t len, uint32_t &height, uint32_t &width)
{
    // set default value, as flag
    height = 0;
    width = 0;

    uint32_t remain_len = len;
    uint32_t sof0_start = 0;
    while (true) {
        if (remain_len < 8) { 
            return;
        }

        for (; sof0_start < len - 1; sof0_start++) {
            if (mjpeg[sof0_start] == 0xFF && mjpeg[sof0_start + 1] == 0xC0) {
                break;
            }
        }

        // can't find sof0
        if (sof0_start == len - 1) {
            return;
        }

        sof0_start += 2;

        if (len - sof0_start < 8) { 
            return;
        }

        height = (mjpeg[sof0_start + 3] << 8) + mjpeg[sof0_start + 4];
        width = (mjpeg[sof0_start + 5] << 8) + mjpeg[sof0_start + 6];

        sof0_start += 8;
        remain_len = len - sof0_start;
    }
}

long SdkInferGetFileSize(const char *fileName)
{
    struct stat statBuf;
    int s32IntRet = stat(fileName, &statBuf);
    if (0 != s32IntRet) {
        LOG_ERROR("[SdkInferGetFileSize] get stat of file[%s] failed, ret %d\n", fileName, s32IntRet);
        return 0;
    }

    long fileSize = statBuf.st_size;

    return fileSize;
}

void SdkInferGetTimeStart(Time_Cost *timeCost, COST_MODULE module)
{
    timeCost->perTime[module][0] = SdkInferElapsedus();
}

void SdkInferGetTimeEnd(Time_Cost *timeCost, COST_MODULE module)
{
    long long costTime;

    timeCost->perTime[module][1] = SdkInferElapsedus();
    costTime = timeCost->perTime[module][1] - timeCost->perTime[module][0];
    timeCost->totalTime[module] += costTime;
    timeCost->totalCount[module] += 1;
}
