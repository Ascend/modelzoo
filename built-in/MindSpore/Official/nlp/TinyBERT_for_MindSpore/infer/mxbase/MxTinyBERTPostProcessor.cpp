#include "MxTinyBERTPostProcessor.h"

APP_ERROR MSTinyBERTPostProcessor::Init(const std::string & /*configPath*/,
                                        const std::string & /*labelPath*/,
                                        MxBase::ModelDesc modelDesc) {
    APP_ERROR ret = APP_ERR_OK;

    this->GetModelTensorsShape(modelDesc);

    ret = CheckMSModelCompatibility();

    if (ret == APP_ERR_OK) {
        ret = ReadConfigParams();
    }
    return ret;
}

/*
 * @description: Do nothing temporarily.
 * @return APP_ERROR error code.
 */
APP_ERROR MSTinyBERTPostProcessor::DeInit() {
    // do nothing for this derived class
    return MxBase::ObjectPostProcessorBase::DeInit();
}

/*
 * @description: Get the info of detected object from output and resize to
 * original coordinates.
 * @param featLayerData Vector of output feature data.
 * @param objInfos Address of output object infos.
 * @param useMpPictureCrop if true, offsets of coordinates will be given.
 * @param postImageInfo Info of model/image width and height, offsets of
 * coordinates.
 * @return: ErrorCode.
 */
APP_ERROR MSTinyBERTPostProcessor::Process(
    std::vector<std::shared_ptr<void>> &featLayerData,
    std::vector<ObjDetectInfo> &objInfos, const bool /*useMpPictureCrop*/,
    MxBase::PostImageInfo /*postImageInfo*/) {
    auto *predict = static_cast<float *>(featLayerData[10].get());
    float c = predict[0] > predict[1] ? 0 : 1;
    ObjDetectInfo objectinfo;
    objectinfo.classId = c;
    objInfos.push_back(objectinfo);
    return APP_ERR_OK;
}
