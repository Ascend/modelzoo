#include "acl/acl.h"
#include "infer_engine.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include <functional>
#include <algorithm>
#include "acl/ops/acl_dvpp.h"
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
    ret = aclInit(configPath);
    CHECK_ACL_RET("acl init failed", ret);

    ret = aclrtSetDevice(cfg.deviceId);
    CHECK_ACL_RET("open device failed ret", ret);

    ret = aclrtCreateContext(&context, cfg.deviceId);
    CHECK_ACL_RET("create context failed", ret);

    cfg.context = context;
    std::cout << "context init done" << std::endl;

    return ACL_ERROR_NONE;
}

aclError UnInitContext()
{
    ret = aclrtDestroyContext(context);
    CHECK_ACL_RET("destory context failed", ret);
    
    ret = aclrtResetDevice(cfg.deviceId);
    CHECK_ACL_RET("reset device failed", ret);

    ret = aclFinalize();
    CHECK_ACL_RET("finalize failed", ret);

    std::cout << "UnInitContext done" << std::endl;
    return ACL_ERROR_NONE;
}

aclError LoadModel()
{
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

    modelDesc = aclmdlCreateDesc();
    CHECK_WITH_RET(modelDesc != nullptr, ACL_ERROR_READ_MODEL_FAILURE, "create model desc failed");
    ret = aclmdlGetDesc(modelDesc, modelId);
    CHECK_ACL_RET("get model desc failed", ret);

    cfg.modelDesc = modelDesc;
    cfg.modelId = modelId;

    std::cout << "LoadModel done" << std::endl;
    return ACL_ERROR_NONE;
}

aclError DvppSetup()
{
    ret = aclrtSetCurrentContext(context);
    CHECK_ACL_RET("set current context failed", ret);

    ret = aclrtCreateStream(&stream);
    CHECK_ACL_RET("create stream failed", ret);

    dvpp_channel_desc = acldvppCreateChannelDesc();
    CHECK_WITH_RET(dvpp_channel_desc != nullptr, ACL_ERROR_DVPP_ERROR, "create dvpp channel desc failed");

    ret = acldvppCreateChannel(dvpp_channel_desc);
    CHECK_ACL_RET("create dvpp channel failed", ret);

    resizedWidthAligned = (resizedWidth + 15) / 16 * 16;
    resizedHeightAligned = (resizedHeight + 1) / 2 * 2;

    resizedOutputBufferSize = resizedWidthAligned * resizedHeightAligned * 3 / 2;
    std::cout << "resizedWidth " << resizedWidth << "resizedHeight " << resizedHeight 
              << "resizedWidthAligned " << resizedWidthAligned << "resizedHeightAligned " << resizedHeightAligned 
              << "resizedHeightAligned " << resizedHeightAligned 
              << std::endl;
    return ACL_ERROR_NONE;
}

acldvppPicDesc *createDvppPicDesc(void *dataDev, acldvppPixelFormat format, uint32_t width, uint32_t height, uint32_t widthStride, uint32_t heightStride, uint32_t size)
{
    acldvppPicDesc *picDesc = acldvppCreatePicDesc();
    CHECK_NULL_RET("create dvpp pic failed", picDesc);

    ret = acldvppSetPicDescData(picDesc, dataDev);
    CHECK_RET_RETURN_NULL("set pic desc failed", ret);

    ret = acldvppSetPicDescSize(picDesc, size);
    CHECK_RET_RETURN_NULL("set pic desc size", ret);

    ret = acldvppSetPicDescFormat(picDesc, format);
    CHECK_RET_RETURN_NULL("set pic desc format failed", ret);

    ret = acldvppSetPicDescWidth(picDesc, width);
    CHECK_RET_RETURN_NULL("set pic desc width failed", ret);

    ret = acldvppSetPicDescHeight(picDesc, height);
    CHECK_RET_RETURN_NULL("set pic desc height failed", ret);

    ret = acldvppSetPicDescWidthStride(picDesc, widthStride);
    CHECK_RET_RETURN_NULL("set pic desc width stride failed", ret);

    ret = acldvppSetPicDescHeightStride(picDesc, heightStride);
    CHECK_RET_RETURN_NULL("set pic desc height stride failed", ret);

    return picDesc;
}

aclError InitInput(std::vector<std::string> files)
{
    ret = aclrtSetCurrentContext(context);
    CHECK_ACL_RET("set current context failed", ret);

    size_t modelInputSize = cfg.inputInfo[0].size;
    size_t imgSize = modelInputSize / cfg.batchSize;

    void *dst;
    ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_ACL_RET("malloc device failed", ret);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();
    for (int i = 0; i < files.size(); i++)
    {

        std::string fileLocation = cfg.dataDir + "/" + files[i];
        FILE *pFile = fopen(fileLocation.c_str(), "r");

        if (pFile == nullptr)
        {
            ret = ACL_ERROR_OTHERS;
            std::cout << "open file " << fileLocation << "failed" << std::endl;
            return ret;
        }

        fseek(pFile, 0, SEEK_END);
        size_t fileSize = ftell(pFile);

        if (fileSize > imgSize)
        {
            ret = ACL_ERROR_OTHERS;
            std::cout << fileLocation << " fileSize " << fileSize 
                      << "* batch " << cfg.batchSize 
                      << " don't match with model inputSize " << modelInputSize 
                      << std::endl;
                     
            return ret;
        }

        void *buff = nullptr;
        ret = aclrtMallocHost(&buff, fileSize);
        CHECK_ACL_RET("malloc host buffer failed", ret);

        rewind(pFile);
        fread(buff, sizeof(char), fileSize, pFile);
        fclose(pFile);

        void *dstTmp = (void *)ptr;
        ret = aclrtMemcpy(dstTmp, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        ptr += fileSize;
        if (ret != ACL_ERROR_NONE)
        {
            std::cout << "init input " << i << ", Copy host to device failed, ret " << ret << std::endl;
            std::cout << "input addr " << dstTmp << ", len " << fileSize << std::endl;
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
        CHECK_ACL_RET("create databuffer failed", ret);
    }

    aclmdlDataset *input = aclmdlCreateDataset();
    ret = aclmdlAddDatasetBuffer(input, inputData);
    if (ret != ACL_ERROR_NONE)
    {
        std::cout << "ACL_ModelInputDataAdd failed, ret " << ret << std::endl;
        aclmdlDestroyDataset(input);
        return ret;
    }

    inputDataframe.dataset = input;
    //std::cout << "init input batch " << processedCnt << " done" << std::endl;
    return ACL_ERROR_NONE;
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
    std::cout << "InitCropRoiConfig right=" << right << ", bottom=" << bottom << std::endl; 
    cropConfig = acldvppCreateRoiConfig(0, right, 0, bottom);
    CHECK_NULL_RET("acldvppCreateRoiConfig failed", cropConfig);

    return cropConfig;
}

acldvppRoiConfig *InitCropCenterRoiConfig(uint32_t newInputWidth, uint32_t newInputHeight, uint32_t modelInputWidth, uint32_t modelInputHeight)
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
    left_half = amount_to_be_cropped_w / 2;
    amount_to_be_cropped_h = newInputHeight - modelInputHeight;
    top_half = amount_to_be_cropped_h / 2;

    left = (left_half % 2 == 0) ? (amount_to_be_cropped_w / 2) : (amount_to_be_cropped_w / 2 + 1);
    top = (top_half % 2 == 0) ? (amount_to_be_cropped_h / 2) : (amount_to_be_cropped_h / 2 + 1);

    right = (left + modelInputWidth - 1) % 2 == 1 ? (left + modelInputWidth - 1) : (left + modelInputWidth - 2);
    bottom = (top + modelInputHeight - 1) % 2 == 1 ? (top + modelInputHeight - 1) : (top + modelInputHeight - 2);

    centerCropConfig = acldvppCreateRoiConfig(left, right, top, bottom);
    printf("left = %d , right = %d , top = %d , bottom = %d\n", left, right, top, bottom);
    CHECK_NULL_RET("acldvppCreateRoiConfig failed", centerCropConfig);

    return centerCropConfig;
}

void GetImageHW(void* buff, uint32_t fileSize, std::string fileLocation, uint32_t &W, uint32_t &H){
    int32_t components = 0;
    acldvppJpegGetImageInfo((void *)buff, fileSize, &W, &H, &components);
}


aclError DVPP_EfficientNetB0(std::string fileLocation, char *&ptr)
{
    /**
     * The preprocessing is divided into two steps:
     * 1. Decode (acldvppJpegDecodeAsync)
     * 2. 80% center crop -> resize (acldvppVpcCropAsync)
    **/

    /**************************Decode**************************/
    uint32_t W, H, W_Aligned, H_Aligned, outputBuffSize;
    void *decodeInput = nullptr;
    void *decodeOutput = nullptr;
    acldvppPicDesc *decodeOutputDesc = nullptr;
    
    //1.0 Prepare the input data of decode
    uint64_t fileSize;
    void *buff = ReadFile(fileLocation, fileSize);
    if( buff == nullptr)
    {
        std::cout << "read file failed" << std::endl;
        return 1;
    }  
    ret = acldvppMalloc(&decodeInput, fileSize);
    CHECK_ACL_RET("malloc dvpp in buff failed", ret);
  
    ret = aclrtMemcpy(decodeInput, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_ACL_RET("copy host to device failed", ret);

    //2.0 Prepare the ouputDesc of decode
    GetImageHW(buff, fileSize, fileLocation, W, H);
    W_Aligned = (W + 127) / 128 * 128;
    H_Aligned = (H + 15) / 16 * 16;
    outputBuffSize = W_Aligned * H_Aligned * 3 / 2;
    ret = acldvppMalloc(&decodeOutput, outputBuffSize);
    CHECK_ACL_RET("malloc decode buff failed", ret);

    decodeOutputDesc = createDvppPicDesc(decodeOutput, PIXEL_FORMAT_YUV_SEMIPLANAR_420, W, H, W_Aligned, H_Aligned, outputBuffSize);
    std::cout << "file[" << fileLocation << "] jpeg picDesc info: W=" << W 
              << ", H=" << H << ", W_Aligned=" << W_Aligned << ", H_Aligned=" << H_Aligned
              << ", outBufSize=" << outputBuffSize << ", format=" << PIXEL_FORMAT_YUV_SEMIPLANAR_420
              <<std::endl;

    //3.0 Decode
    ret = acldvppJpegDecodeAsync(dvpp_channel_desc, decodeInput, fileSize, decodeOutputDesc, stream);
    CHECK_ACL_RET("acldvppJpegDecodeAsync failed", ret);

    aclrtFreeHost(buff);
    aclrtSynchronizeStream(stream);

    /**************************Center crop**************************/
    acldvppRoiConfig *centralcropConfig = nullptr;
    acldvppPicDesc *centralcropOutputDesc = nullptr; // resize output desc
    float central_fraction = 0.875;
    uint32_t smallerSide = W < H ? W : H;
    uint32_t centralcropWidth = smallerSide * central_fraction;
    uint32_t centralcropHeight = smallerSide * central_fraction;

    centralcropConfig = InitCropCenterRoiConfig(W, H, centralcropWidth, centralcropHeight);

    void *dstTmp = (void *)ptr;
    centralcropOutputDesc = createDvppPicDesc(dstTmp, PIXEL_FORMAT_YUV_SEMIPLANAR_420, resizedWidth, resizedHeight, resizedWidthAligned, resizedHeightAligned, resizedOutputBufferSize);
    if (centralcropOutputDesc == nullptr)
    {
        std::cout << "create cropOutputDesc failed" << std::endl;
        return 1;
    }

    ret = acldvppVpcCropAsync(dvpp_channel_desc, decodeOutputDesc, centralcropOutputDesc, centralcropConfig, stream);
    CHECK_ACL_RET("acldvppVpcCropAsync failed", ret);

    ptr += resizedOutputBufferSize;
    aclrtSynchronizeStream(stream);
  
    /*************************Release resources************************/
    acldvppFree(decodeInput);
    acldvppFree(decodeOutput);
    acldvppDestroyPicDesc(decodeOutputDesc);
    acldvppDestroyPicDesc(centralcropOutputDesc);
    acldvppDestroyRoiConfig(centralcropConfig);

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

    //malloc memory for model input
    void *dst;
    ret = acldvppMalloc(&dst, cfg.inputInfo[0].size);
    CHECK_ACL_RET("malloc device failed", ret);

    char *ptr = (char *)dst;
    inputDataframe.fileNames.clear();
    
    for (int i = 0; i < files.size(); i++)
    {
        std::string fileLocation = cfg.dataDir + "/" + files[i];
        ret = DVPP_EfficientNetB0(fileLocation, ptr);
        CHECK_ACL_RET("DVPP_EfficientNetB0 failed", ret);
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
        std::cout << "aclCreateDataBuffer failed" << std::endl;
        return 1;
    }

    ret = aclmdlAddDatasetBuffer(input, inputData);
    CHECK_ACL_RET("aclmdlAddDatasetBuffer failed", ret);

    inputDataframe.dataset = input;
    return ACL_ERROR_NONE;
}

aclError Inference()
{
    ret = aclrtSetCurrentContext(context);
    CHECK_ACL_RET("set infer context failed", ret);

    struct timeval startTmp, endTmp;
    long long timeUse;

    if (inputDataframe.fileNames.size() == 0)
    {
        ret = ACL_ERROR_OTHERS;
        std::cout << "No file found" << std::endl;
        return ret;
    }

    aclmdlDataset *output = aclmdlCreateDataset();
    if (output == nullptr)
    {
        ret = ACL_ERROR_OTHERS;
        std::cout << "Create Output Dataset failed" << std::endl;
        return ret;
    }

    std::vector<void *> outputDevPtrs;

    for (size_t i = 0; i < cfg.outputNum; ++i)
    {
        size_t buffer_size = cfg.outputInfo[i].size;
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, (size_t)buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);

        CHECK_ACL_RET("malloc output host buff failed", ret);
        outputDevPtrs.push_back(outputBuffer);
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, buffer_size);

        if (outputData == nullptr)
        {
            ret = ACL_ERROR_OTHERS;
            std::cout << "Create output data buffer failed" << std::endl;
            return ret;
        }

        ret = aclmdlAddDatasetBuffer(output, outputData);
        CHECK_ACL_RET("Add output model dataset failed", ret);
    }

    gettimeofday(&startTmp, NULL);
    ret = aclmdlExecute(modelId, inputDataframe.dataset, output);
    gettimeofday(&endTmp, NULL);
    timeUse = (endTmp.tv_sec - startTmp.tv_sec) * 1000000 + (endTmp.tv_usec - startTmp.tv_usec);
    std::cout << inputDataframe.fileNames[0] << " inference time use: " << timeUse << " us" << std::endl;
    inferTime += timeUse;

    if (ret != ACL_ERROR_NONE)
    {
        std::cout << inputDataframe.fileNames[0] << " inference failed." << std::endl;
        FreeDevMemory(inputDataframe.dataset);
        aclmdlDestroyDataset(inputDataframe.dataset);
        return ret;
    }

    outputDataframe.fileNames = inputDataframe.fileNames;
    outputDataframe.dataset = output;

    uint32_t dvppFlag;
    (cfg.useDvpp) ? dvppFlag = 1 : dvppFlag = 0;

    ret = DestroyDatasetResurce(inputDataframe.dataset, dvppFlag);
    CHECK_ACL_RET("DestroyDatasetResurce failed", ret);
    std::cout << "inference batch " << processedCnt << " done" << std::endl;
    return ACL_ERROR_NONE;
}

aclError UnloadModel()
{
    ret = aclmdlUnload(modelId);
    CHECK_ACL_RET("unload model failed", ret);

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
