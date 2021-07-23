/* *
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* */

#include "acl/acl.h"
#include "infer_engine.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/ops/acl_dvpp.h"
#include <functional>
#include <algorithm>
using namespace std;

std::unordered_map<std::string, long long> dvppTime;
extern Resnet50Result resnet50Res;
extern Config cfg;
extern aclError ret;
extern int processedCnt;
extern long long inferTime;
aclrtContext context;
uint32_t modelId;
aclmdlDesc *modelDesc;
std::vector<std::string> files;
DataFrame inputDataframe;
DataFrame outputDataframe;
aclDataBuffer *yoloImgInfo;
aclrtStream stream = nullptr;
acldvppChannelDesc *dvpp_channel_desc = nullptr;
std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> imgSizes;

#define RESIZE_MIN 256
#define NUM_2 2
#define NUM_3 3
#define NUM_16 16
#define NUM_128 128

uint32_t resizedWidth;
uint32_t resizedHeight;
uint32_t resizedWidthAligned;
uint32_t resizedHeightAligned;
uint32_t resizedOutputBufferSize;

void getImgResizeShape()
{
    if (ACL_FORMAT_NCHW == cfg.inputInfo[0].Format) {
        resizedHeight = cfg.inputInfo[0].dims[NUM_2];
        resizedWidth = cfg.inputInfo[0].dims[NUM_3];
    } else if (ACL_FORMAT_NHWC == cfg.inputInfo[0].Format) {
        resizedHeight = cfg.inputInfo[0].dims[1];
        resizedWidth = cfg.inputInfo[0].dims[NUM_2];
    }
    return;
}

aclError InitContext(const char *configPath)
{
    LOG("context init start\n");
    ret = aclInit(configPath);
    CHECK_ACL_RET("acl init failed", ret);

    ret = aclrtSetDevice(cfg.deviceId);
    CHECK_ACL_RET("open device failed ret", ret);

    ret = aclrtCreateContext(&context, cfg.deviceId);
    CHECK_ACL_RET("create context failed", ret);

    cfg.context = context;
    LOG("context init done\n");
    return ACL_ERROR_NONE;
}

aclError UnInitContext()
{
    ret = aclrtDestroyContext(context);
    CHECK_ACL_RET("destory context failed", ret);
    LOG("destory context done\n");

    ret = aclrtResetDevice(cfg.deviceId);
    CHECK_ACL_RET("reset device failed", ret);

    ret = aclFinalize();
    CHECK_ACL_RET("finalize failed", ret);
    LOG("reset device done\n");

    return ACL_ERROR_NONE;
}

aclError LoadModel()
{
    LOG("load model start\n");
    size_t memSize;
    size_t weightsize;
    uint32_t modelSize = 0;
    std::string modelPath = cfg.om;

    cfg.modelData_ptr = ReadBinFile(modelPath, modelSize);
    CHECK_WITH_RET(cfg.modelData_ptr != nullptr, ACL_ERROR_READ_MODEL_FAILURE, "can't read model");

    aclError ret = aclmdlQuerySizeFromMem(cfg.modelData_ptr, modelSize, &memSize, &weightsize);
    CHECK_ACL_RET("query memory size failed", ret);

    ret = aclrtMalloc(&(cfg.devMem_ptr), memSize, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK_ACL_RET("alloc dev_ptr failed", ret);
    ret = aclrtMalloc(&(cfg.weightMem_ptr), weightsize, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK_ACL_RET("alloc weight_ptr failed", ret);

    ret = aclmdlLoadFromMemWithMem(cfg.modelData_ptr, modelSize, &modelId, cfg.devMem_ptr, memSize, cfg.weightMem_ptr,
                                   weightsize);
    CHECK_ACL_RET("load model from memory failed", ret);
    LOG("Load model success. memSize: %lu, weightSize: %lu.\n", memSize, weightsize);

    modelDesc = aclmdlCreateDesc();
    CHECK_WITH_RET(modelDesc != nullptr, ACL_ERROR_READ_MODEL_FAILURE, "create model desc failed");
    ret = aclmdlGetDesc(modelDesc, modelId);
    CHECK_ACL_RET("get model desc failed", ret);

    cfg.modelDesc = modelDesc;
    cfg.modelId = modelId;

    LOG("load model done\n");
    return ACL_ERROR_NONE;
}

aclError DvppSetup()
{
    ret = aclrtSetCurrentContext(context);
    if (ret != ACL_ERROR_NONE) {
        LOG("Set context failed\n");
        return ret;
    }

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        LOG("create dvpp stream failed\n");
        return ret;
    }

    dvpp_channel_desc = acldvppCreateChannelDesc();
    if (dvpp_channel_desc == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("create dvpp channel desc failed\n");
        return ret;
    }

    ret = acldvppCreateChannel(dvpp_channel_desc);
    if (ret != ACL_ERROR_NONE) {
        LOG("create dvpp channel failed\n");
        return ret;
    }

    imgSizes = cfg.dvppConfig.imgSizes;

    resizedWidthAligned = (resizedWidth + 15) / NUM_16 * NUM_16;
    resizedHeightAligned = (resizedHeight + 1) / NUM_2 * NUM_2;

    resizedOutputBufferSize = resizedWidthAligned * resizedHeightAligned * NUM_3 / NUM_2;
    LOG("resizedWidth %d resizedHeight %d resizedWidthAligned %d resizedHeightAligned %d resizedOutputBufferSize %d\n",
        resizedWidth, resizedHeight, resizedWidthAligned, resizedHeightAligned, resizedOutputBufferSize);

    return ACL_ERROR_NONE;
}

/*
 * @brief : 生成dvpp图像描述信息
 * @param [in] void *dataDev : 码流buffer信息.
 * @param [in] acldvppPixelFormat format: 图像格式
 * @param [in] uint32_t width : 宽度
 * @param [in] uint32_t height: 高度
 * @param [in] uint32_t widthStride : 宽度对齐.
 * @param [in] uint32_t heightStride: 高度对齐.
 * @param [in] uint32_t size: 码流大小.
 * @return : acldvppPicDesc：图像描述信息
 */
acldvppPicDesc *createDvppPicDesc(void *dataDev, acldvppPixelFormat format, uint32_t width, uint32_t height,
                                  uint32_t widthStride, uint32_t heightStride, uint32_t size)
{
    acldvppPicDesc *picDesc = acldvppCreatePicDesc();
    if (picDesc == nullptr) {
        LOG("failed to create pic desc\n");
        return nullptr;
    }

    ret = acldvppSetPicDescData(picDesc, dataDev);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc data\n");
        return nullptr;
    }
    ret = acldvppSetPicDescSize(picDesc, size);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc size\n");
        return nullptr;
    }

    ret = acldvppSetPicDescFormat(picDesc, format);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc format\n");
        return nullptr;
    }

    ret = acldvppSetPicDescWidth(picDesc, width);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc width\n");
        return nullptr;
    }

    ret = acldvppSetPicDescHeight(picDesc, height);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc height\n");
        return nullptr;
    }

    ret = acldvppSetPicDescWidthStride(picDesc, widthStride);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc widthStride\n");
        return nullptr;
    }

    ret = acldvppSetPicDescHeightStride(picDesc, heightStride);
    if (ret != ACL_ERROR_NONE) {
        LOG("failed to set desc heightStride\n");
        return nullptr;
    }
    return picDesc;
}

aclError InitInput(std::vector<std::string> files)
{
    LOG("init input batch %d start\n", processedCnt);
    ret = aclrtSetCurrentContext(context);
    if (ret != ACL_ERROR_NONE) {
        LOG("Set context failed, ret[%d]\n", ret);
        return ret;
    }

    size_t modelInputSize = cfg.inputInfo[0].size;
    size_t imgSize = modelInputSize / cfg.batchSize;

    void *dst;
    ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc device failed, ret[%d]\n", ret);
        return ret;
    }
    LOG("dst = %p, size = %ld\n", dst, modelInputSize);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();
    for (int i = 0; i < files.size(); i++) {

        std::string fileLocation = cfg.dataDir + "/" + files[i];
        FILE *pFile = fopen(fileLocation.c_str(), "r");

        if (pFile == nullptr) {
            ret = ACL_ERROR_OTHERS;
            LOG("open file %s failed\n", fileLocation.c_str());
            return ret;
        }

        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);

        if (fileSize > imgSize) {
            ret = ACL_ERROR_OTHERS;
            LOG("%s fileSize %lu * batch %lu don't match with model inputSize %lu\n", fileLocation.c_str(),
                fileSize, cfg.batchSize, modelInputSize);
            return ret;
        }

        void *buff = nullptr;
        ret = aclrtMallocHost(&buff, fileSize);
        if (ret != ACL_ERROR_NONE) {
            LOG("Malloc host buff failed[%d]\n", ret);
            return ret;
        }

        rewind(pFile);
        fread(buff, sizeof(char), fileSize, pFile);
        fclose(pFile);

        void *dstTmp = (void *)ptr;
        ret = aclrtMemcpy(dstTmp, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        ptr += fileSize;
        LOG("input addr %p, len %ld\n", dstTmp, fileSize);
        if (ret != ACL_ERROR_NONE) {
            LOG("init input %d, Copy host to device failed, ret[%d]\n", i, ret);
            LOG("input addr %p, len %ld\n", dstTmp, fileSize);
            aclrtFreeHost(buff);
            return ret;
        }

        aclrtFreeHost(buff);
        inputDataframe.fileNames.push_back(files[i]);
    }

    aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, modelInputSize);
    if (inputData == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("aclCreateDataBuffer failed\n");
        return ret;
    }

    aclmdlDataset *input = aclmdlCreateDataset();
    ret = aclmdlAddDatasetBuffer(input, inputData);
    if (ret != ACL_ERROR_NONE) {
        LOG("ACL_ModelInputDataAdd failed, ret[%d]\n", ret);
        aclmdlDestroyDataset(input);
        return ret;
    }

    inputDataframe.dataset = input;
    LOG("init input batch %d done\n", processedCnt);
    return ACL_ERROR_NONE;
}

/*
 * @brief : 获取图像宽高
 * @param [in] void* buff : 输入码流地址.
 * @param [in] uint32_t fileSize : 输入码流长度
 * @param [in] std::string fileLocation : 输入文件路径.
 * @param [in] uint32_t &W : 输入码流宽度.
 * @param [in] uint32_t &H : 输入码流高度.
 */
void GetImageHW(void* buff, uint32_t fileSize, std::string fileLocation, uint32_t &W, uint32_t &H)
{
    int32_t components = 0;
    ret = acldvppJpegGetImageInfo((void *)buff, fileSize, &W, &H, &components);
    if (ret != ACL_ERROR_NONE) {
        cout << "acldvppJpegGetImageInfo failed, ret " << ret << "filename: " << fileLocation.c_str() << endl;
    }
}

/*
 * @brief : dvpp在推理中的预处理流程
 * @param [in] string fileLocation : 输入文件路径.
 * @param [in] char *&ptr : 输出buffer指针.
 * @return : ACL_ERROR_NONE：预处理成功， 其他：预处理失败
 */
aclError DVPP_Resnet50(std::string fileLocation, char *&ptr)
{
    // 1 获取输入码流
    uint32_t W, H, W_Aligned, H_Aligned, outputBuffSize;
    void *decodeInput = nullptr;
    void *decodeOutput = nullptr;
    acldvppPicDesc *decodeOutputDesc = nullptr;
    uint64_t fileSize;
    void *buff = ReadFile(fileLocation, fileSize);
    if( buff == nullptr) {
        LOG("read pic failed\n");
        return 1;
    }

    ret = acldvppMalloc(&decodeInput, fileSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc dvpp in buff failed[%d]\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(decodeInput, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        LOG("copy host to device failed[%d]\n", ret);
        return ret;
    }

    // 2 获取解码输出描述信息
    GetImageHW(buff, fileSize, fileLocation, W, H);
    W_Aligned = (W + 127) / NUM_128 * NUM_128;
    H_Aligned = (H + 15) / NUM_16 * NUM_16;
    outputBuffSize = W_Aligned * H_Aligned * NUM_3 / NUM_2;
    ret = acldvppMalloc(&decodeOutput, outputBuffSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc decodeOutput buff failed[%d]\n", ret);
        return ret;
    }
    decodeOutputDesc = createDvppPicDesc(decodeOutput, PIXEL_FORMAT_YUV_SEMIPLANAR_420, W, H, W_Aligned, H_Aligned,
                                         outputBuffSize);
    if (decodeOutputDesc == nullptr) {
        LOG("create jpeg_output_desc failed\n");
        return 1;
    }
    LOG("file[%s] jpeg picDesc info: W=%d, H=%d, W_Aligned=%d, H_Aligned=%d, outBufSize=%d, format=%d\n", \ 
        fileLocation.c_str(),W, H, W_Aligned, H_Aligned, outputBuffSize, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    // 3 使用jpegd图像解码
    ret = acldvppJpegDecodeAsync(dvpp_channel_desc, decodeInput, fileSize, decodeOutputDesc, stream);
    if (ret != ACL_ERROR_NONE) {
        LOG(" dvppJpegDecodeAsync failed\n");
        return ret;
    }
    aclrtFreeHost(buff);
    aclrtSynchronizeStream(stream);

    // 4 对jpegd解码的图片进行原分辨率抠图及短边256等比例缩放
    acldvppRoiConfig *cropConfig = nullptr;
    acldvppPicDesc *cropOutputDesc = nullptr;
    // 设置对解码后的图片进行原图裁剪，目的是为了减少因jpegd解码后对齐的无效数据对图像精度的影响
    cropConfig = InitCropRoiConfig(W, H);

    uint32_t newInputWidth = 0;
    uint32_t newInputHeight = 0;
    void *cropOutBufferDev = nullptr;
    // 宽和高较短的一条边缩放至256，较长边做等比例缩放。对齐至256目的是为了给224x224中心抠图做准备,短边256对齐，获得对齐后的宽高
    SmallSizeAtLeast(W, H, newInputWidth, newInputHeight);
    uint32_t cropOutputWidthStride = (newInputWidth + (NUM_16 - 1)) / NUM_16 * NUM_16;
    uint32_t cropOutputHeightStride = (newInputHeight + (NUM_2 - 1)) / NUM_2 * NUM_2;
    uint32_t cropOutBufferSize = cropOutputWidthStride * cropOutputHeightStride * NUM_3 / NUM_2;
    ret = acldvppMalloc(&cropOutBufferDev, cropOutBufferSize);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "[ERROR][Vision] AcldvppMalloc cropOutBufferDev_ failed, ret = " << ret << " cropOutBufferSize_ = "
                  << cropOutBufferSize << endl;
        return ret;
    }
    cropOutputDesc = createDvppPicDesc(cropOutBufferDev, PIXEL_FORMAT_YUV_SEMIPLANAR_420, newInputWidth, newInputHeight,
                                       cropOutputWidthStride, cropOutputHeightStride, cropOutBufferSize);
    if (cropOutputDesc == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("create cropOutputDesc failed\n");
        return ret;
    }

    ret = acldvppVpcCropAsync(dvpp_channel_desc, decodeOutputDesc, cropOutputDesc, cropConfig, stream);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "[ERROR][Vision] acldvppVpcCropAsync failed, ret = " << ret << std::endl;
        return ret;
    }
    aclrtSynchronizeStream(stream);

    // 5 对等比例缩放后的图片做224x224中心抠图，中心抠图后的数据会发送给aipp进行YUV转RGB格式转换。需要注意：中心抠图后的输出格式和aipp
    // 的输入格式需要保持一致。
    acldvppRoiConfig *centerCropConfig = nullptr;
    acldvppPicDesc *centerCropOutputDesc = nullptr; // resize output desc
    centerCropConfig = InitCropCenterRoiConfig(newInputWidth, newInputHeight, resizedWidth, resizedHeight);
    void *vpcOutBufferDev = nullptr;
    uint32_t vpcOutBufferSize = resizedWidthAligned * resizedHeightAligned * NUM_3 / NUM_2;

    vpcOutBufferDev = (void *)ptr;
    centerCropOutputDesc = createDvppPicDesc(vpcOutBufferDev, PIXEL_FORMAT_YUV_SEMIPLANAR_420, resizedWidth,
                                             resizedHeight, resizedWidthAligned, resizedHeightAligned,
                                             vpcOutBufferSize);
    if (centerCropOutputDesc == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("create centerCropOutputDesc failed\n");
        return ret;
    }

    ret = acldvppVpcCropAsync(dvpp_channel_desc, cropOutputDesc, centerCropOutputDesc, centerCropConfig, stream);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "[ERROR][Vision] acldvppVpcCropAsync failed, ret = " << ret << "fileName: " << fileLocation.c_str() << std::endl;
        return ret;
    }

    ptr += vpcOutBufferSize;
    aclrtSynchronizeStream(stream);
 
    // 6 释放资源
    acldvppFree(decodeInput);
    acldvppFree(decodeOutput);
    acldvppFree(cropOutBufferDev);
    acldvppDestroyPicDesc(decodeOutputDesc);
    acldvppDestroyPicDesc(cropOutputDesc);
    acldvppDestroyPicDesc(centerCropOutputDesc);
    acldvppDestroyRoiConfig(cropConfig);
    acldvppDestroyRoiConfig(centerCropConfig);
    return ret;
}

aclError DvppInitInput(std::vector<std::string> files)
{
    struct timeval process_start;
    struct timeval process_end;
    std::string funcName;
    long long costTime;
    funcName = "DvppTotalProcess";
    gettimeofday(&process_start, NULL);

    void *dst;
    ret = acldvppMalloc(&dst, cfg.inputInfo[0].size);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc device failed, ret[%d]\n", ret);
        return ret;
    }

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();

    for (int i = 0; i < files.size(); i++) {
        std::string fileLocation = cfg.dataDir + "/" + files[i];
        ret = DVPP_Resnet50(fileLocation, ptr);
        if(ret != ACL_ERROR_NONE) {
            LOG("dvpp config failed");
            return ret;
        }
        inputDataframe.fileNames.push_back(files[i]);
    }

    funcName = "DvppTotalProcess";
    gettimeofday(&process_end, NULL);
    costTime = (process_end.tv_sec - process_start.tv_sec) * 1000000 + (process_end.tv_usec - process_start.tv_usec);
    dvppTime[funcName] += costTime;

    aclmdlDataset *input = aclmdlCreateDataset();
    aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, cfg.inputInfo[0].size);

    if (inputData == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("aclCreateDataBuffer failed\n");
        return ret;
    }

    ret = aclmdlAddDatasetBuffer(input, inputData);

    if (ret != ACL_ERROR_NONE) {
        LOG("ACL_ModelInputDataAdd failed, ret[%d]\n", ret);
        aclmdlDestroyDataset(input);
        return ret;
    }

    inputDataframe.dataset = input;
    return ACL_ERROR_NONE;
}

acldvppRoiConfig *InitCropRoiConfig(uint32_t width, uint32_t height)
{
    uint32_t right = 0;
    uint32_t bottom = 0;
    acldvppRoiConfig *cropConfig;

    if (width % NUM_2 == 0) {
        right = width - 1;
    } else {
        right = width;
    }

    if (height % NUM_2 == 0) {
        bottom = height - 1;
    } else {
        bottom = height;
    }

    cropConfig = acldvppCreateRoiConfig(0, right, 0, bottom);
    if (cropConfig == nullptr) {
        std::cout << "[ERROR][Vision] acldvppCreateRoiConfig failed " << std::endl;
        return nullptr;
    }

    return cropConfig;
}

acldvppRoiConfig *InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight, uint32_t modelInputWidth,
                                          uint32_t modelInputHeight)
{
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t top = 0;
    uint32_t bottom = 0;
    uint32_t amount_to_be_cropped_w = 0;
    uint32_t amount_to_be_cropped_h = 0;
    uint32_t left_half = 0;
    uint32_t top_half = 0;
    acldvppRoiConfig *centerCropConfig = nullptr;

    // 计算中心抠图起始点的坐标距离码流左边界和上边界的距离
    amount_to_be_cropped_w = newInputWidth - modelInputWidth;
    left_half = amount_to_be_cropped_w / NUM_2;
    amount_to_be_cropped_h = newInputHeight - modelInputHeight;
    top_half = amount_to_be_cropped_h / NUM_2;

    // 保证起始点坐标为偶数
    left = (left_half % NUM_2 == 0) ? (amount_to_be_cropped_w / NUM_2) : (amount_to_be_cropped_w / NUM_2 + 1);
    top = (top_half % NUM_2 == 0) ? (amount_to_be_cropped_h / NUM_2) : (amount_to_be_cropped_h / NUM_2 + 1);

    // 结束点为奇数
    right = left + modelInputWidth - 1;
    bottom = top + modelInputHeight - 1;

    centerCropConfig = acldvppCreateRoiConfig(left, right, top, bottom);
    if (centerCropConfig == nullptr) {
        std::cout << "[ERROR][Vision] acldvppCreateRoiConfig failed " << std::endl;
        return nullptr;
    }
    return centerCropConfig;
}

void SmallSizeAtLeast(uint32_t width, uint32_t height, uint32_t &newInputWidth, uint32_t &newInputHeight)
{
    float scaleRatio = 0.0;
    float inputWidth = 0.0;
    float inputHeight = 0.0;
    float resizeMin = 0.0;
    bool minWidthFlag = false;

    inputWidth = (float)width;
    inputHeight = (float)height;
    resizeMin = (float)(RESIZE_MIN);
    minWidthFlag = (width <= height) ? true : false;

    // 短边缩放为256，长边等比例缩放
    if (minWidthFlag == true) {
        newInputWidth = resizeMin;
        newInputHeight = (resizeMin / width) * inputHeight;
        std::cout << "[INFO]scaleRatio: " << resizeMin / width << " inputWidth_: " << width << " newInputWidth: " <<
            newInputWidth << " inputHeight_: " << inputHeight << " newInputHeight_:" << newInputHeight << std::endl;
    } else {
        newInputWidth = (resizeMin / height) * width;
        newInputHeight = resizeMin;
        std::cout << "[INFO]scaleRatio: " << resizeMin / height << " width: " << width << " newInputWidth: " <<
            newInputWidth << " height: " << height << " newInputHeight:" << newInputHeight << std::endl;
    }
}

aclError Inference()
{
    LOG("inference batch %d start\n", processedCnt);
    ret = aclrtSetCurrentContext(context);

    if (ret != ACL_ERROR_NONE) {
        LOG("Set infer context failed\n");
        return ret;
    }

    struct timeval startTmp, endTmp;
    long long timeUse;

    if (inputDataframe.fileNames.size() == 0) {
        ret = ACL_ERROR_OTHERS;
        LOG("No file found\n");
        return ret;
    }

    aclmdlDataset *output = aclmdlCreateDataset();
    if (output == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("Create Output Dataset failed\n");
        return ret;
    }

    std::vector<void *> outputDevPtrs;

    for (size_t i = 0; i < cfg.outputNum; ++i) {
        size_t buffer_size = cfg.outputInfo[i].size;
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, (size_t)buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);

        if (ret != ACL_ERROR_NONE) {
            LOG("Malloc output host failed, ret[%d]\n", ret);
            return ret;
        }

        outputDevPtrs.push_back(outputBuffer);
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, buffer_size);

        if (outputData == nullptr) {
            ret = ACL_ERROR_OTHERS;
            LOG("Create output data buffer failed\n");
            return ret;
        }

        ret = aclmdlAddDatasetBuffer(output, outputData);

        if (ret != ACL_ERROR_NONE) {
            LOG("Add output model dataset failed, ret[%d]\n", ret);
            return ret;
        }
    }

    gettimeofday(&startTmp, NULL);
    ret = aclmdlExecute(modelId, inputDataframe.dataset, output);
    gettimeofday(&endTmp, NULL);
    timeUse = (endTmp.tv_sec - startTmp.tv_sec) * 1000000 + (endTmp.tv_usec - startTmp.tv_usec);
    LOG("%s inference time use: %lld us\n", inputDataframe.fileNames[0].c_str(), timeUse);
    inferTime += timeUse;

    if (ret != ACL_ERROR_NONE) {
        LOG("%s inference failed.\n", inputDataframe.fileNames[0].c_str());
        FreeDevMemory(inputDataframe.dataset);
        aclmdlDestroyDataset(inputDataframe.dataset);
        return ret;
    }

    outputDataframe.fileNames = inputDataframe.fileNames;
    outputDataframe.dataset = output;

    uint32_t dvppFlag = (cfg.useDvpp) ? 1 : 0;
    ret = DestroyDatasetResurce(inputDataframe.dataset, dvppFlag);
    if (ret != ACL_ERROR_NONE) {
        LOG("DestroyDatasetResurce failed\n");
        return ret;
    }

    LOG("inference batch %d done\n", processedCnt);
    return ACL_ERROR_NONE;
}

aclError UnloadModel()
{
    LOG("unload model start\n");
    ret = aclmdlUnload(modelId);
    CHECK_ACL_RET("unload model failed", ret);
    LOG("unload model done\n");

    aclmdlDestroyDesc(cfg.modelDesc);

    if (cfg.devMem_ptr != nullptr) {
        aclrtFree(cfg.devMem_ptr);
        cfg.devMem_ptr = nullptr;
    }

    if (cfg.weightMem_ptr != nullptr) {
        aclrtFree(cfg.weightMem_ptr);
        cfg.weightMem_ptr = nullptr;
    }

    if (cfg.modelData_ptr != nullptr) {
        delete[] cfg.modelData_ptr;
        cfg.modelData_ptr = nullptr;
    }
    return ACL_ERROR_NONE;
}
