#include "acl/acl.h"
#include "infer_engine.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include <functional>
#include <algorithm>
//#include <opencv2/opencv.hpp>
#include "acl/ops/acl_dvpp.h"
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

uint32_t resizedWidth;
uint32_t resizedHeight;
uint32_t resizedWidthAligned;
uint32_t resizedHeightAligned;
uint32_t resizedOutputBufferSize;

void getImgResizeShape()
{
    if (ACL_FORMAT_NCHW == cfg.inputInfo[0].Format)
    {
        resizedHeight = cfg.inputInfo[0].dims[2];
        resizedWidth = cfg.inputInfo[0].dims[3];
    }
    else if (ACL_FORMAT_NHWC == cfg.inputInfo[0].Format)
    {
        resizedHeight = cfg.inputInfo[0].dims[1];
        resizedWidth = cfg.inputInfo[0].dims[2];
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

    ret = aclmdlLoadFromMemWithMem(cfg.modelData_ptr, modelSize, &modelId, cfg.devMem_ptr, memSize, cfg.weightMem_ptr, weightsize);
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
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Set context failed\n");
        return ret;
    }

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("create dvpp stream failed\n");
        return ret;
    }

    dvpp_channel_desc = acldvppCreateChannelDesc();
    if (dvpp_channel_desc == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("create dvpp channel desc failed\n");
        return ret;
    }

    ret = acldvppCreateChannel(dvpp_channel_desc);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("create dvpp channel failed\n");
        return ret;
    }

    resizedWidthAligned = (resizedWidth + 15) / 16 * 16;
    resizedHeightAligned = (resizedHeight + 1) / 2 * 2;

    resizedOutputBufferSize = resizedWidthAligned * resizedHeightAligned * 3 / 2;
    LOG("resizedWidth %d resizedHeight %d resizedWidthAligned %d resizedHeightAligned %d resizedOutputBufferSize %d\n", resizedWidth, resizedHeight, resizedWidthAligned, resizedHeightAligned, resizedOutputBufferSize);

    return ACL_ERROR_NONE;
}

acldvppPicDesc *createDvppPicDesc(void *dataDev, acldvppPixelFormat format, uint32_t width, uint32_t height, uint32_t widthStride, uint32_t heightStride, uint32_t size)
{
    acldvppPicDesc *picDesc = acldvppCreatePicDesc();
    if (picDesc == nullptr)
    {
        LOG("failed to create pic desc\n");
        return nullptr;
    }

    ret = acldvppSetPicDescData(picDesc, dataDev);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc data\n");
        return nullptr;
    }
    ret = acldvppSetPicDescSize(picDesc, size);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc size\n");
        return nullptr;
    }

    ret = acldvppSetPicDescFormat(picDesc, format);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc format\n");
        return nullptr;
    }

    ret = acldvppSetPicDescWidth(picDesc, width);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc width\n");
        return nullptr;
    }

    ret = acldvppSetPicDescHeight(picDesc, height);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc height\n");
        return nullptr;
    }

    ret = acldvppSetPicDescWidthStride(picDesc, widthStride);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc widthStride\n");
        return nullptr;
    }

    ret = acldvppSetPicDescHeightStride(picDesc, heightStride);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("failed to set desc heightStride\n");
        return nullptr;
    }
    return picDesc;
}

aclError InitInput(std::vector<std::string> files)
{
    LOG("init input batch %d start\n", processedCnt);
    ret = aclrtSetCurrentContext(context);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Set context failed, ret[%d]\n", ret);
        return ret;
    }

    size_t modelInputSize = cfg.inputInfo[0].size;
    size_t imgSize = modelInputSize / cfg.batchSize;

    void *dst;
    ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Malloc device failed, ret[%d]\n", ret);
        return ret;
    }
    LOG("dst = %p, size = %ld\n", dst, modelInputSize);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();
    for (int i = 0; i < files.size(); i++)
    {

        std::string fileLocation = cfg.dataDir + "/" + files[i];
        FILE *pFile = fopen(fileLocation.c_str(), "r");

        if (pFile == nullptr)
        {
            ret = ACL_ERROR_OTHERS;
            LOG("open file %s failed\n", fileLocation.c_str());
            return ret;
        }

        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);

        if (fileSize > imgSize)
        {
            ret = ACL_ERROR_OTHERS;
            LOG("%s fileSize %lu * batch %lu don't match with model inputSize %lu\n", fileLocation.c_str(), fileSize, cfg.batchSize, modelInputSize);
            return ret;
        }

        void *buff = nullptr;
        ret = aclrtMallocHost(&buff, fileSize);
        if (ret != ACL_ERROR_NONE)
        {
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
        if (ret != ACL_ERROR_NONE)
        {
            LOG("init input %d, Copy host to device failed, ret[%d]\n", i, ret);
            LOG("input addr %p, len %ld\n", dstTmp, fileSize);

            aclrtFreeHost(buff);
            return ret;
        }

        aclrtFreeHost(buff);

        inputDataframe.fileNames.push_back(files[i]);
    }

    aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, modelInputSize);
    if (inputData == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("aclCreateDataBuffer failed\n");
        return ret;
    }

    aclmdlDataset *input = aclmdlCreateDataset();
    ret = aclmdlAddDatasetBuffer(input, inputData);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("ACL_ModelInputDataAdd failed, ret[%d]\n", ret);
        aclmdlDestroyDataset(input);
        return ret;
    }

    inputDataframe.dataset = input;
    LOG("init input batch %d done\n", processedCnt);
    return ACL_ERROR_NONE;
}

aclError GetImageSize(std::string file)
{
    std::string fileLocation = cfg.dataDir + "/" + file;
    FILE *pFile = fopen(fileLocation.c_str(), "r");

    if (pFile == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("open file %s failed\n", fileLocation.c_str());
        return ret;
    }

    fseek(pFile, 0, SEEK_END);
    size_t fileSize = ftell(pFile);

    void *buff = nullptr;
    ret = aclrtMallocHost(&buff, fileSize);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Malloc host buff failed[%d]\n", ret);
        return ret;
    }

    rewind(pFile);
    fread(buff, sizeof(char), fileSize, pFile);
    fclose(pFile);

    uint32_t width, height;
    int32_t components;
    ret = acldvppJpegGetImageInfo(buff, fileSize, &width, &height, &components);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("acldvppJpegGetImageInfo failed");
        aclrtFreeHost(buff);
    }
    printf("==============fileName=%s, W=%d, H=%d===============\n", file.c_str(), width, height);
    auto size = std::make_pair(width, height);
    imgSizes[file] = std::move(size);

    aclrtFreeHost(buff);

    return ret;
}

acldvppRoiConfig *InitCropRoiConfig(uint32_t width, uint32_t height)
{
    uint32_t right = 0;
    uint32_t bottom = 0;
    acldvppRoiConfig *cropConfig;

    if (width % 2 == 0)
    {
        right = width - 1;
    }
    else
    {
        right = width;
    }

    if (height % 2 == 0)
    {
        bottom = height - 1;
    }
    else
    {
        bottom = height;
    }
    printf("InitCropRoiConfig right=%d, bottom=%d \n", right, bottom);
    cropConfig = acldvppCreateRoiConfig(0, right, 0, bottom);
    if (cropConfig == nullptr)
    {
        std::cout << "[ERROR][Vision] acldvppCreateRoiConfig failed " << std::endl;
        return nullptr;
    }

    return cropConfig;
}

acldvppRoiConfig *InitCropCenterRoiConfig(uint32_t InputWidth, uint32_t InputHeight, uint32_t cropInputWidth, uint32_t cropInputHeight)
{
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t top = 0;
    uint32_t bottom = 0;
    uint32_t amount_to_be_cropped_w = 0;
    uint32_t amount_to_be_cropped_h = 0;
    uint32_t left_half = 0;
    uint32_t top_half = 0;
    printf("CentralCrop InputWidth=%d InputHeight=%d cropInputWidth=%d cropInputHeight=%d \n", InputWidth, InputHeight, cropInputWidth, cropInputHeight);

    acldvppRoiConfig *centerCropConfig = nullptr;

    amount_to_be_cropped_w = InputWidth - cropInputWidth;
    left_half = amount_to_be_cropped_w / 2;
    amount_to_be_cropped_h = InputHeight - cropInputHeight;
    top_half = amount_to_be_cropped_h / 2;

    left = (left_half % 2 == 0) ? (amount_to_be_cropped_w / 2) : (amount_to_be_cropped_w / 2 + 1);
    top = (top_half % 2 == 0) ? (amount_to_be_cropped_h / 2) : (amount_to_be_cropped_h / 2 + 1);

    right = left + cropInputWidth - 1;
    bottom = top + cropInputHeight - 1;

    centerCropConfig = acldvppCreateRoiConfig(left, right, top, bottom);
    if (centerCropConfig == nullptr)
    {
        std::cout << "[ERROR][Vision] InitCropCenterRoiConfig acldvppCreateRoiConfig failed " << std::endl;
        return nullptr;
    }

    return centerCropConfig;
}

void GetImageHW(void* buff, uint32_t fileSize, std::string fileLocation, uint32_t &W, uint32_t &H){
    int32_t components = 0;
    ret = acldvppJpegGetImageInfo((void *)buff, fileSize, &W, &H, &components);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "acldvppJpegGetImageInfo failed, ret " << ret << "filename: " << fileLocation.c_str() << endl;
    }
}

void* ReadFile(std::string fileLocation, uint64_t &fileSize)
{
	aclError ret;
	FILE *pFile = fopen(fileLocation.c_str(), "r");
    if (pFile == nullptr)
    {
		std::cout << "open file " << fileLocation << " failed" << std::endl;
        return nullptr;
    }

    fseek(pFile, 0, SEEK_END);
    fileSize = ftell(pFile);

    void *buff = nullptr;
    ret = aclrtMallocHost(&buff, fileSize);
	if(ret != ACL_ERROR_NONE)
    {
        cout << "aclrtMallocHost failed" << endl;
    }

    rewind(pFile);
    fread(buff, sizeof(char), fileSize, pFile);
    fclose(pFile);
	return buff;
}

aclError DVPP_Densenet121(std::string fileLocation, char *&ptr)
{
    uint32_t W, H, W_Aligned, H_Aligned, outputBuffSize;
    void *decodeInput = nullptr;
    void *decodeOutput = nullptr;
    acldvppPicDesc *decodeOutputDesc = nullptr;
    
    //1.0 Prepare the input data of decode
    uint64_t fileSize;
    void *buff = ReadFile(fileLocation, fileSize);
    if( buff == nullptr)
    {
        LOG("read pic failed");
        return 1;
    }  
    ret = acldvppMalloc(&decodeInput, fileSize);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Malloc dvpp in buff failed[%d]\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(decodeInput, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("copy host to device failed[%d]\n", ret);
        return ret;
    }
    
    //2.0 Prepare the ouputDesc of decode
    GetImageHW(buff, fileSize, fileLocation, W, H);
    W_Aligned = (W + 127) / 128 * 128;
    H_Aligned = (H + 15) / 16 * 16;
    
    outputBuffSize = W_Aligned * H_Aligned * 3 / 2;


    ret = acldvppMalloc(&decodeOutput, outputBuffSize);
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Malloc decodeOutput buff failed[%d]\n", ret);
        return ret;
    }
    decodeOutputDesc = createDvppPicDesc(decodeOutput, PIXEL_FORMAT_YUV_SEMIPLANAR_420, W, H, W_Aligned, H_Aligned, outputBuffSize);
    if (decodeOutputDesc == nullptr)
    {
        LOG("create decodeOutputDesc failed\n");
        return 1;
    }
    LOG("file[%s] jpeg picDesc info: W=%d, H=%d, W_Aligned=%d, H_Aligned=%d, outBufSize=%d, format=%d\n", \ 
                fileLocation.c_str(),W, H, W_Aligned, H_Aligned, outputBuffSize, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    //3.0 Decode
    ret = acldvppJpegDecodeAsync(dvpp_channel_desc, decodeInput, fileSize, decodeOutputDesc, stream);
    if (ret != ACL_ERROR_NONE)
    {
        LOG(" dvppJpegDecodeAsync failed\n");
        return ret;
    }
    aclrtFreeHost(buff);
    aclrtSynchronizeStream(stream);

    
    //Crop original image and Resize [256, 256]
    acldvppRoiConfig *cropConfig = nullptr;
    acldvppPicDesc *cropOutputDesc = nullptr; // resize output desc
    cropConfig = InitCropRoiConfig(W, H);
    void *cropOutBufferDev = nullptr;
    uint32_t cropOutBufferSize = 256 * 256 * 3 / 2;
    ret = acldvppMalloc(&cropOutBufferDev, cropOutBufferSize);
    if (ret != ACL_ERROR_NONE)
    {
        std::cout << "[ERROR][Vision] AcldvppMalloc cropOutBufferDev_ failed, ret = " << ret << " cropOutBufferSize_ = " << cropOutBufferSize << endl;
        return ret;
    }
    cropOutputDesc = createDvppPicDesc(cropOutBufferDev, PIXEL_FORMAT_YUV_SEMIPLANAR_420, 256, 256, 256, 256, cropOutBufferSize);
    if (cropOutputDesc == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("create cropOutputDesc failed\n");
        return ret;
    }
    ret = acldvppVpcCropAsync(dvpp_channel_desc, decodeOutputDesc, cropOutputDesc, cropConfig, stream);
    if (ret != ACL_ERROR_NONE)
    {
        std::cout << "[ERROR][Vision] crop acldvppVpcCropAsync failed, ret = " << ret << std::endl;
        return ret;
    }

    //Central crop [224 224]
    aclrtSynchronizeStream(stream);
    acldvppRoiConfig *centralcropConfig = nullptr;
    acldvppPicDesc *centralcropOutputDesc = nullptr; // resize output desc
    centralcropConfig = InitCropCenterRoiConfig(256, 256, resizedWidth, resizedHeight);
    void *dstTmp = (void *)ptr;
    centralcropOutputDesc = createDvppPicDesc(dstTmp, PIXEL_FORMAT_YUV_SEMIPLANAR_420, resizedWidth, resizedHeight, resizedWidthAligned, resizedHeightAligned, resizedOutputBufferSize);
    if (cropOutputDesc == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("create cropOutputDesc failed\n");
        return ret;
    }
    ptr += resizedOutputBufferSize;
    ret = acldvppVpcCropAsync(dvpp_channel_desc, cropOutputDesc, centralcropOutputDesc, centralcropConfig, stream);
    if (ret != ACL_ERROR_NONE)
    {
        std::cout << "[ERROR][Vision] acldvppVpcCropAsync failed, ret = " << ret << std::endl;
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);

    acldvppFree(decodeInput);
    acldvppFree(decodeOutput);
    acldvppFree(cropOutBufferDev);
    acldvppDestroyPicDesc(decodeOutputDesc);
    acldvppDestroyPicDesc(cropOutputDesc);
    acldvppDestroyPicDesc(centralcropOutputDesc);
    acldvppDestroyRoiConfig(cropConfig);
    acldvppDestroyRoiConfig(centralcropConfig);

    return ACL_ERROR_NONE;
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
    if (ret != ACL_ERROR_NONE)
    {
        LOG("Malloc device failed, ret[%d]\n", ret);
        return ret;
    }

    LOG("DvppInitInput dvpp malloc dst size:%d\n", cfg.inputInfo[0].size);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();

    for (int i = 0; i < files.size(); i++)
    {

        //ret = GetImageSize(files[i]);
        std::string fileLocation = cfg.dataDir + "/" + files[i];
        ret = DVPP_Densenet121(fileLocation, ptr);
        inputDataframe.fileNames.push_back(files[i]);
    }

    funcName = "DvppTotalProcess";
    gettimeofday(&process_end, NULL);
    costTime = (process_end.tv_sec - process_start.tv_sec) * 1000000 + (process_end.tv_usec - process_start.tv_usec);
    dvppTime[funcName] += costTime;

    aclmdlDataset *input = aclmdlCreateDataset();
    aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, cfg.inputInfo[0].size);

    if (inputData == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("aclCreateDataBuffer failed\n");
        return ret;
    }

    ret = aclmdlAddDatasetBuffer(input, inputData);

    if (ret != ACL_ERROR_NONE)
    {
        LOG("ACL_ModelInputDataAdd failed, ret[%d]\n", ret);
        aclmdlDestroyDataset(input);
        return ret;
    }

    inputDataframe.dataset = input;
    return ACL_ERROR_NONE;
}

aclError Inference()
{
    LOG("inference batch %d start\n", processedCnt);
    ret = aclrtSetCurrentContext(context);

    if (ret != ACL_ERROR_NONE)
    {
        LOG("Set infer context failed\n");
        return ret;
    }

    struct timeval startTmp, endTmp;
    long long timeUse;

    if (inputDataframe.fileNames.size() == 0)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("No file found\n");
        return ret;
    }

    aclmdlDataset *output = aclmdlCreateDataset();
    if (output == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        LOG("Create Output Dataset failed\n");
        return ret;
    }

    std::vector<void *> outputDevPtrs;

    for (size_t i = 0; i < cfg.outputNum; ++i)
    {
        size_t buffer_size = cfg.outputInfo[i].size;
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, (size_t)buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        //LOG("NN output addr %ld, len %ld ",size_t(outputBuffer), size_t(buffer_size));

        if (ret != ACL_ERROR_NONE)
        {
            LOG("Malloc output host failed, ret[%d]\n", ret);
            return ret;
        }
        //LOG("output%ld: addr %ld, size %ld\n", i, size_t(outputBuffer) , size_t(buffer_size));
        outputDevPtrs.push_back(outputBuffer);
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, buffer_size);

        if (outputData == nullptr)
        {
            ret = ACL_ERROR_OTHERS;
            LOG("Create output data buffer failed\n");
            return ret;
        }

        ret = aclmdlAddDatasetBuffer(output, outputData);

        if (ret != ACL_ERROR_NONE)
        {
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

    if (ret != ACL_ERROR_NONE)
    {
        LOG("%s inference failed.\n", inputDataframe.fileNames[0].c_str());
        FreeDevMemory(inputDataframe.dataset);
        aclmdlDestroyDataset(inputDataframe.dataset);
        return ret;
    }

    outputDataframe.fileNames = inputDataframe.fileNames;
    outputDataframe.dataset = output;

    uint32_t dvppFlag;
    (cfg.useDvpp) ? dvppFlag = 1 : dvppFlag = 0;

    ret = DestroyDatasetResurce(inputDataframe.dataset, dvppFlag);
    if (ret != ACL_ERROR_NONE)
    {
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

    if (nullptr != cfg.devMem_ptr)
    {
        aclrtFree(cfg.devMem_ptr);
        cfg.devMem_ptr = nullptr;
    }

    if (nullptr != cfg.weightMem_ptr)
    {
        aclrtFree(cfg.weightMem_ptr);
        cfg.weightMem_ptr = nullptr;
    }

    if (nullptr != cfg.modelData_ptr)
    {
        delete[] cfg.modelData_ptr;
        cfg.modelData_ptr = nullptr;
    }
    return ACL_ERROR_NONE;
}
