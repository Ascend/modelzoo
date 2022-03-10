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
aclrtStream stream = nullptr;
acldvppChannelDesc *dvpp_channel_desc = nullptr;
std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> imgSizes;

#define RESIZE_MAX 416
#define NUM_2 2
uint32_t resizedWidth;
uint32_t resizedHeight;
uint32_t resizedWidthAligned;
uint32_t resizedHeightAligned;
uint32_t resizedOutputBufferSize;

std::ofstream outFile("img_info", std::ios::trunc);

void getImgResizeShape()
{
    if (ACL_FORMAT_NCHW == cfg.inputInfo[0].Format) {
        resizedHeight = cfg.inputInfo[0].dims[NUM_2];
        resizedWidth = cfg.inputInfo[0].dims[3];
    } else if (ACL_FORMAT_NHWC == cfg.inputInfo[0].Format) {
        resizedHeight = cfg.inputInfo[0].dims[1];
        resizedWidth = cfg.inputInfo[0].dims[NUM_2];
    }
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

    resizedWidthAligned = (resizedWidth + 15) / 16 * 16;
    resizedHeightAligned = (resizedHeight + 1) / NUM_2 * NUM_2;

    resizedOutputBufferSize = resizedWidthAligned * resizedHeightAligned * 3 / NUM_2;
    LOG("resizedWidth %d resizedHeight %d resizedWidthAligned %d resizedHeightAligned %d resizedOutputBufferSize %d\n",
        resizedWidth, resizedHeight, resizedWidthAligned, resizedHeightAligned, resizedOutputBufferSize);

    return ACL_ERROR_NONE;
}

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
        std::string fileLocation = files[i];
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
            LOG("%s fileSize %lu * batch %lu don't match with model inputSize %lu\n", fileLocation.c_str(), fileSize,
                cfg.batchSize, modelInputSize);
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
 * @brief : 设置贴图区域
 * @param [in] uint32_t width : 贴入图像的宽.
 * @param [in] uint32_t height : 贴入图像的高.
 * @param [in] uint32_t modelInputWidth : 贴图后图像的宽.
 * @param [in] uint32_t modelInputHeight : 贴图后图像的高.
 * @return : acldvppRoiConfig：贴图配置
 */
acldvppRoiConfig *InitVpcOutConfig(uint32_t width, uint32_t height, uint32_t modelInputWidth, uint32_t modelInputHeight)
{
    uint32_t right = 0;
    uint32_t bottom = 0;
    uint32_t left = 0;
    uint32_t top = 0;
    uint32_t left_stride;
    acldvppRoiConfig *cropConfig;

    uint32_t small = width < height ? width : height;
    uint32_t padded_size_half;
    char tmpChr[256] = {0};

    if (small == width) {
        padded_size_half = (modelInputWidth - width) / NUM_2; // 贴图区域距离左边界的距离
        left = padded_size_half;
        left_stride = (left + 15) / 16 * 16;
        right = (left_stride + width) % NUM_2 == 0 ? (left_stride + width - 1) : (left_stride + width);
        if (left_stride + right > modelInputWidth) {
            while (true) {
                left_stride = left_stride - 16;
                right = (left_stride + width) % NUM_2 == 0 ? (left_stride + width - 1) : (left_stride + width);
                if (left_stride + right < modelInputWidth)
                    break;
            }
        }

        right = (left_stride + width) % NUM_2 == 0 ? (left_stride + width - 1) : (left_stride + width);
        bottom = (modelInputHeight % NUM_2 == 0 ? modelInputHeight - 1 : modelInputHeight);
        top = bottom - height + 1;
    } else {
        padded_size_half = (modelInputHeight - height) / NUM_2;
        right = (modelInputWidth % NUM_2 == 0 ? modelInputWidth - 1 : modelInputWidth);
        left = right + 1 - width;
        left_stride = (left + 15) / 16 * 16;
        top = (padded_size_half % NUM_2 == 0 ? padded_size_half : padded_size_half + 1);
        bottom = (height + top - 1) % NUM_2 == 0 ? (height + top - NUM_2) : (height + top - 1);
    }

    LOG("left_stride=%d, right=%d, top=%d, bottom=%d\n", left_stride, right, top, bottom);

    cropConfig = acldvppCreateRoiConfig(left_stride, right, top, bottom);
    if (cropConfig == nullptr) {
        std::cout << "[ERROR][Vision] acldvppCreateRoiConfig failed " << std::endl;
        return nullptr;
    }
    snprintf(tmpChr, sizeof(tmpChr), "%d %d\n", left_stride, top);
    outFile << tmpChr;

    return cropConfig;
}

void LargeSizeAtLeast(uint32_t W, uint32_t H, uint32_t &newInputWidth, uint32_t &newInputHeight)
{
    float scaleRatio = 0.0;
    float inputWidth = 0.0;
    float inputHeight = 0.0;
    float resizeMax = 0.0;
    bool maxWidthFlag = false;
    inputWidth = (float)W;
    inputHeight = (float)H;
    resizeMax = (float)(RESIZE_MAX);
    maxWidthFlag = (W >= H) ? true : false;

    char tmpChr[256] = {0};
    if (maxWidthFlag == true) {
        newInputWidth = resizeMax;
        scaleRatio = resizeMax / W;
        // 高度2对齐
        newInputHeight = scaleRatio * inputHeight;
        newInputHeight = (newInputHeight + 1) / NUM_2 * NUM_2;
        std::cout << "[info]scaleRatio: " << resizeMax / W << " inputWidth_: " << W << " newInputWidth: " <<
             newInputWidth << " inputHeight_: " << H << " newInputHeight_:" << newInputHeight << std::endl;
    } else {
        scaleRatio = resizeMax / H;
        // 如果高度是长边，建议宽度在等比例缩放后再做一次16对齐。因为vpc在输出时宽有16字节对齐约束，当贴图的宽非16对齐时，会导致在贴图的时候，
        // 芯片会自动进行16字节对齐，导致每次写入数据的时候都会引入部分无效数据，从而导致精度下降。
        newInputWidth = scaleRatio * W;
        newInputWidth = (newInputWidth + 15) / 16 * 16;
        newInputHeight = resizeMax;
        std::cout << "[info]scaleRatio: " << resizeMax / H << " width: " << W << " newInputWidth: " << newInputWidth <<
             " height: " << H << " newInputHeight:" << newInputHeight << std::endl;
    }
    snprintf(tmpChr, sizeof(tmpChr), "%f %f ", (float)newInputWidth / W, (float)newInputHeight / H);
    outFile << tmpChr;
}

/*
 * @brief : dvpp在YOLOv3推理中的预处理流程
 * @param [in] string fileLocation : 输入文件路径.
 * @param [in] char *&ptr : 输出buffer指针.
 * @return : ACL_ERROR_NONE：预处理成功， 其他：预处理失败
 */
aclError DVPP_Yolo(std::string fileLocation, char *&ptr)
{
    struct timeval func_start;
    struct timeval func_end;
    
    std::string funcName;
    long long costTime;
    // 1 获取输入码流
    FILE *pFile = fopen(fileLocation.c_str(), "r");
    if (pFile == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("open file %s failed\n", fileLocation.c_str());
        return ret;
    }

    fseek(pFile, 0, SEEK_END);
    uint64_t fileSize = ftell(pFile);
    void *buff = nullptr;
    ret = aclrtMallocHost(&buff, fileSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc host buff failed[%d]\n", ret);
        return ret;
    }

    rewind(pFile);
    fread(buff, sizeof(char), fileSize, pFile);
    fclose(pFile);

    uint32_t W = 0;
    uint32_t H = 0;
    uint32_t W_Aligned = 0;
    uint32_t H_Aligned = 0;
    uint32_t outputBuffSize = 0;
    int32_t components = 0;

    // 2 获取输入解码宽高
    ret = acldvppJpegGetImageInfo((void *)buff, fileSize, &W, &H, &components);
    if (ret != ACL_ERROR_NONE) {
        cout << "acldvppJpegGetImageInfo failed, ret " << ret << "filename: " << fileLocation.c_str() << endl;
    }

#ifdef ASCEND710_DVPP
    W_Aligned = (W + 63) / 64 * 64;
    H_Aligned = (H + 15) / 16 * 16;
    if(W_Aligned > 4096 || H_Aligned > 4096){
        return -1;
    }
    acldvppJpegFormat realformat;
    int aclformat;
    acldvppJpegGetImageInfoV2(buff, fileSize, &W, &H, &components,&realformat);
    switch (realformat){
        case 0:
            aclformat = 6;
            outputBuffSize = W_Aligned * H_Aligned * 3;
            break;
        case 1:
            aclformat = 4;
            outputBuffSize = W_Aligned * H_Aligned * 2;
            break;
        case 2:
            aclformat = 2;
            outputBuffSize = W_Aligned * H_Aligned  * 3/2;
            break;
        case 4:
            aclformat = 1001;
            outputBuffSize = W_Aligned * H_Aligned  * 2;
            break;
        case 3:
            aclformat = 0;
            outputBuffSize = W_Aligned * H_Aligned;
            break;    
        default:
            aclformat = 1;
            outputBuffSize = W_Aligned * H_Aligned * 3/2;
            break;
    }
    if (aclformat == 0) {
        aclformat = 1;
        outputBuffSize = outputBuffSize * 3/2;
    }    
#else
    W_Aligned = (W + 127) / 128 * 128;
    H_Aligned = (H + 15) / 16 * 16;
    outputBuffSize = W_Aligned * H_Aligned * 3 / NUM_2;
#endif

    void *jpeg_dev_mem_in_ptr = nullptr;
    ret = acldvppMalloc(&jpeg_dev_mem_in_ptr, fileSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc dvpp in buff failed[%d]\n", ret);
        return ret;
    }

    // 把jpeg码流从host侧拷贝到device侧
    ret = aclrtMemcpy(jpeg_dev_mem_in_ptr, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        LOG("copy host to device failed[%d]\n", ret);
        return ret;
    }

    aclrtFreeHost(buff);
    // 申请device侧输出内存
    void *jpeg_dev_mem_out_ptr = nullptr;
    ret = acldvppMalloc(&jpeg_dev_mem_out_ptr, outputBuffSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc dvpp in buff failed[%d]\n", ret);
        return ret;
    }
    ret = aclrtMemset(jpeg_dev_mem_out_ptr, outputBuffSize, 128, outputBuffSize);

    acldvppPicDesc *jpeg_output_desc = nullptr;
    acldvppPicDesc *resize_output_desc = nullptr;
    funcName = "DvppPicDescCreate_output";
    gettimeofday(&func_start, NULL);

    // 3 获取解码输出描述信息
    jpeg_output_desc = createDvppPicDesc(jpeg_dev_mem_out_ptr, PIXEL_FORMAT_YUV_SEMIPLANAR_420, W, H, W_Aligned,
                                         H_Aligned, outputBuffSize);
    if (jpeg_output_desc == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("create jpeg_output_desc failed\n");
        return ret;
    }
    gettimeofday(&func_end, NULL);

    LOG("file[%s] jpeg picDesc info: W=%d, H=%d, W_Aligned=%d, H_Aligned=%d, outBufSize=%d, format=%d\n",
        fileLocation.c_str(), W, H, W_Aligned, H_Aligned, outputBuffSize, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    costTime = (func_end.tv_sec - func_start.tv_sec) * 1000000 + (func_end.tv_usec - func_start.tv_usec);
    dvppTime[funcName] += costTime;

    funcName = "DvppJpegDecode";
    gettimeofday(&func_start, NULL);
    ret = acldvppJpegDecodeAsync(dvpp_channel_desc, jpeg_dev_mem_in_ptr, fileSize, jpeg_output_desc, stream);
    if (ret != ACL_ERROR_NONE) {
        LOG(" dvppJpegDecodeAsync failed, fileName: %s\n", fileLocation.c_str());
        return ret;
    }
    gettimeofday(&func_end, NULL);
    costTime = (func_end.tv_sec - func_start.tv_sec) * 1000000 + (func_end.tv_usec - func_start.tv_usec);
    dvppTime[funcName] += costTime;
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        LOG(" aclrtSynchronizeStream failed, acldvppJpegDecodeAsync \n");
        return ret;
    }
	ret = acldvppGetPicDescRetCode(jpeg_output_desc);
    if (ret != ACL_ERROR_NONE)
	{
        printf(" acldvppGetPicDescRetCode failed\n");
        return ret;
    }

    // 4 对jpegd解码的图片进行原分辨率抠图及长边416等比例缩放。
    acldvppPicDesc *cropOutputDesc = nullptr;
    acldvppRoiConfig *cropConfig = nullptr;
    // 设置对解码后的图片进行原图裁剪，目的是为了减少因jpegd解码后对齐的无效数据对图像精度的影响
#ifdef ASCEND710_DVPP
	uint32_t w_new = acldvppGetPicDescWidth(jpeg_output_desc);
    uint32_t h_new = acldvppGetPicDescHeight(jpeg_output_desc);
    uint32_t format = acldvppGetPicDescFormat(jpeg_output_desc);
    W = w_new;
    H = h_new;
    printf("w_new=%d, h_new=%d, format=%u\n", w_new, h_new, format);
#endif
    cropConfig = InitCropRoiConfig(W, H);
    uint32_t newInputWidth = 0;
    uint32_t newInputHeight = 0;
    void *cropOutputBufferDev = nullptr;
    // 对宽高较长缩放至416，较短边做等比例缩放
    LargeSizeAtLeast(W, H, newInputWidth, newInputHeight);

    uint32_t cropOutputWidthStride = (newInputWidth + (16 - 1)) / 16 * 16;
    uint32_t cropOutputHeightStride = (newInputHeight + (NUM_2 - 1)) / NUM_2 * NUM_2;
    uint32_t cropOutBufferSize = cropOutputWidthStride * cropOutputHeightStride * 3 / NUM_2;
    ret = acldvppMalloc(&cropOutputBufferDev, cropOutBufferSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("acldvppMalloc cropOutputBufferDev failed ret = %d\n", ret);
        return ret;
    }

    ret = aclrtMemset(cropOutputBufferDev, cropOutBufferSize, 128, cropOutBufferSize);
    cropOutputDesc = createDvppPicDesc(cropOutputBufferDev, PIXEL_FORMAT_YUV_SEMIPLANAR_420, newInputWidth,
                                       newInputHeight, cropOutputWidthStride, cropOutputHeightStride,
                                       cropOutBufferSize);
    if (cropOutputDesc == nullptr) {
        ret = ACL_ERROR_OTHERS;
        LOG("create cropOutputDesc failed\n");
    }

    // 原格式抠图以及长边等比例缩放可以在一个接口中完成
#ifdef ASCEND710_DVPP
    acldvppResizeConfig *resizeConfig = acldvppCreateResizeConfig();
	ret = acldvppSetResizeConfigInterpolation(resizeConfig, 1);
	if (ret != ACL_ERROR_NONE)
    {
        std::cout << "[ERROR][Vision] resize acldvppSetResizeConfigInterpolatio failed, ret = " << ret << std::endl;
        return ret;
    }
	ret = acldvppVpcCropResizeAsync(dvpp_channel_desc, jpeg_output_desc, cropOutputDesc, cropConfig, resizeConfig, stream);
#else
    ret = acldvppVpcCropAsync(dvpp_channel_desc, jpeg_output_desc, cropOutputDesc, cropConfig, stream);
#endif
    
    if (ret != ACL_ERROR_NONE) {
        LOG("acldvppVpcCropAsync failed, ret=%d\n", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        LOG(" aclrtSynchronizeStream failed, acldvppVpcCropResizeAsync \n");
        return ret;
    }
    acldvppDestroyRoiConfig(cropConfig);
    cropConfig = nullptr;

    // 5 使用vpc裁剪并贴图
    acldvppPicDesc *cropAndPasteOutputDesc = nullptr;
    acldvppRoiConfig *pasteConfig = nullptr;
    uint32_t vpcOutBufferSize = 416 * 416 * 3 / NUM_2;
    uint32_t vpcOutputWidthStride = (416 + 15) / 16 * 16;
    uint32_t vpcOutputHeightStride = (416 + 1) / NUM_2 * NUM_2;

    void *vpcOutBufferDev = (void *)ptr;
    cropAndPasteOutputDesc = createDvppPicDesc(vpcOutBufferDev, PIXEL_FORMAT_YUV_SEMIPLANAR_420, 416, 416,
                                               vpcOutputWidthStride, vpcOutputHeightStride, vpcOutBufferSize);
    cropConfig = InitCropRoiConfig(newInputWidth, newInputHeight);
    // 设置贴图区域以及贴图目标区域
    pasteConfig = InitVpcOutConfig(newInputWidth, newInputHeight, 416, 416);
#ifdef ASCEND710_DVPP
    ret = acldvppVpcCropResizePasteAsync(dvpp_channel_desc, cropOutputDesc, cropAndPasteOutputDesc, cropConfig, pasteConfig, resizeConfig, stream);
#else
    ret = acldvppVpcCropAndPasteAsync(dvpp_channel_desc, cropOutputDesc, cropAndPasteOutputDesc, cropConfig,pasteConfig, stream);
#endif

    if (ret != ACL_ERROR_NONE) {
        LOG("acldvppVpcCropAndPasteAsync failed, ret = %d\n", ret);
        return ret;
    }

    gettimeofday(&func_end, NULL);
    costTime = (func_end.tv_sec - func_start.tv_sec) * 1000000 + (func_end.tv_usec - func_start.tv_usec);
    dvppTime[funcName] += costTime;

    funcName = "StreamSynchronize";
    gettimeofday(&func_start, NULL);
    ptr += vpcOutBufferSize;
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE)
	{
        LOG(" aclrtSynchronizeStream failed, acldvppVpcCropResizePasteAsync\n");
        return ret;
    }
    gettimeofday(&func_end, NULL);
    costTime = (func_end.tv_sec - func_start.tv_sec) * 1000000 + (func_end.tv_usec - func_start.tv_usec);
    dvppTime[funcName] += costTime;
    if (ret != ACL_ERROR_NONE) {
        LOG("dvpp invoke failed.ret=%d fileName: %s\n", ret, fileLocation.c_str());
        return ret;
    }

    ret = acldvppFree(jpeg_dev_mem_in_ptr);
    if (ret != ACL_ERROR_NONE) {
        LOG("jpeg_dev_mem_in_ptr free failed\n");
        return ret;
    }

    ret = acldvppFree(jpeg_dev_mem_out_ptr);
    if (ret != ACL_ERROR_NONE) {
        LOG("jpeg_dev_mem_out_ptr free failed\n");
    }

    ret = acldvppFree(cropOutputBufferDev);
    if (ret != ACL_ERROR_NONE) {
        LOG("cropOutBufferDev free failed\n");
    }

    // 6 释放资源
    acldvppDestroyPicDesc(jpeg_output_desc);
    acldvppDestroyPicDesc(cropOutputDesc);
    acldvppDestroyPicDesc(cropAndPasteOutputDesc);
    acldvppDestroyRoiConfig(pasteConfig);
    acldvppDestroyRoiConfig(cropConfig);
    return ACL_ERROR_NONE;
}

aclError DvppInitInput(std::vector<std::string> files)
{
    struct timeval process_start;
    struct timeval process_end;
    long long costTime;
    std::string funcName;

    funcName = "DvppTotalProcess";
    gettimeofday(&process_start, NULL);
    void *dst;
    aclError ret;
    ret = acldvppMalloc(&dst, cfg.inputInfo[0].size);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc device failed, ret[%d]\n", ret);
        return ret;
    }

    aclrtMemset(dst, cfg.inputInfo[0].size, 128, cfg.inputInfo[0].size);
    LOG("DvppInitInput dvpp malloc dst size:%d\n", cfg.inputInfo[0].size);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();
    char tmpChr[256] = {0};

    for (int i = 0; i < files.size(); i++) {
        snprintf(tmpChr, sizeof(tmpChr), "%s ", files[i].c_str());
        outFile << tmpChr;
        std::string fileLocation = files[i];
        ret = DVPP_Yolo(fileLocation, ptr);
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
#ifdef ASCEND710_DVPP
    right = width - 1;
    bottom = height - 1;
#else
    if ((width % NUM_2) == 0) {
        right = width - 1;
    } else {
        right = width;
    }

    if ((height % NUM_2) == 0) {
        bottom = height - 1;
    } else {
        bottom = height;
    }
#endif

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