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
#include "multi_inputs_inference_engine.h"
using namespace std;

multi_inputs_Inference_engine::multi_inputs_Inference_engine() : InferEngine() {}

multi_inputs_Inference_engine::multi_inputs_Inference_engine(Config *config) : InferEngine(config) {}

multi_inputs_Inference_engine::multi_inputs_Inference_engine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
    : InferEngine(inQue, outQue)
{}
aclError multi_inputs_Inference_engine::CreateInferInput(std::vector<std::string> &inferFile_vec)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t pos = 0;
    aclDataBuffer *inputData;
    void *dst;

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        LOG_ERROR("aclmdlCreateDataset failed, ret[%d]", ret);
        aclrtFree(dst);
        aclDestroyDataBuffer(inputData);
        return 1;
    }
    for (int j = 0; j < cfg_->inputArray.size(); j++) {
        uint32_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, j);
        uint32_t singleImgSize = modelInputSize / maxBatch_;

        ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Malloc device failed, ret[%d]", ret);
            return ret;
        }

        inputData = aclCreateDataBuffer((void *)dst, modelInputSize);
        if (inputData == nullptr) {
            LOG_ERROR("aclCreateDataBuffer failed");
            aclrtFree(dst);
            return 1;
        }
        pos = 0;
        for (int i = 0; i < inferFile_vec.size(); i++) {
            std::string fileLocation = cfg_->inputArray[j] + "/" + inferFile_vec[i];
            FILE *pFile = fopen(fileLocation.c_str(), "r");
            if (pFile == nullptr) {
                LOG_ERROR("[ERROR]open file %s failed", fileLocation.c_str());
                continue;
            }

            LOG_INFO("[INFO]load img index %d, img file name %s success", i, fileLocation.c_str());

            fseek(pFile, 0, SEEK_END);
            size_t fileSize = ftell(pFile);
            rewind(pFile);
            if (fileSize > singleImgSize) {
                LOG_ERROR("[ERROR]%s fileSize %ld can not more than model each batch inputSize %d",
                    fileLocation.c_str(), fileSize, singleImgSize);
                fclose(pFile);
                continue;
            }

            if (runMode_ == ACL_HOST) {
                void *buff = nullptr;
                ret = aclrtMallocHost(&buff, fileSize);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]Malloc host buff failed[%d]", ret);
                    fclose(pFile);
                    continue;
                }

                fread(buff, sizeof(char), fileSize, pFile);
                fclose(pFile);

                ret = aclrtMemcpy((uint8_t *)dst + pos, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]Copy host to device failed, ret[%d]", ret);
                    aclrtFreeHost(buff);
                    continue;
                }
                pos += fileSize;
                aclrtFreeHost(buff);
            } else {
                fread((uint8_t *)dst + pos, sizeof(char), fileSize, pFile);
                fclose(pFile);
                pos += fileSize;
            }
        }

        ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlAddDatasetBuffer failed, ret[%d]", ret);
            aclrtFree(dst);
            aclDestroyDataBuffer(inputData);
            aclmdlDestroyDataset(input_);
            return ret;
        }
    }
    return ret;
}


void Multi_Inputs_InferThread(multi_inputs_Inference_engine *inferEngine)
{
    aclError ret = aclrtSetCurrentContext(inferEngine->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return;
    }
    LOG_INFO("[INFO]aclrtSetCurrentContext success");

    int cnt = 0;
    int batchSize = inferEngine->cfg_->batchSize;
    uint32_t loopCnt = 0;
    std::vector<std::string> inferFile_vec;

    while (loopCnt < inferEngine->cfg_->loopNum) {
        for (int i = 0; i < inferEngine->files_.size(); i++) {
            if (cnt % batchSize == 0) {
                inferFile_vec.clear();
            }

            cnt++;
            inferFile_vec.push_back(inferEngine->files_[i]);

            if (cnt % batchSize == 0) {
                ret = inferEngine->CreateInferInput(inferFile_vec);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]CreateInferInput failed, ret %d", ret);
                    continue;
                }

                ret = inferEngine->CreateInferOutput();
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]CreateInferOutput failed, ret %d", ret);
                    inferEngine->DestroyDataset(inferEngine->input_);
                    continue;
                }

                ret = inferEngine->ExecInference();
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]ExecInference failed");
                    inferEngine->DestroyDataset(inferEngine->input_);
                    inferEngine->DestroyDataset(inferEngine->output_);
                    continue;
                }

                if (ret == ACL_ERROR_NONE) {
                    int ret32 = inferEngine->SaveInferResult(inferEngine->output_, &inferFile_vec);
                    if (ret != 0) {
                        LOG_ERROR("[ERROR]SaveInferResult failed, ret %d", ret32);
                    }

                    if (inferEngine->cfg_->isDynamicAipp == true) {
                        aclmdlDestroyAIPP(inferEngine->aippDynamicSet_);
                    }
                    inferEngine->DestroyDataset(inferEngine->input_);
                    inferEngine->DestroyDataset(inferEngine->output_);
                }
            }
        }

        loopCnt++;
    }
    if (cnt % batchSize != 0) {
        ret = inferEngine->CreateInferInput(inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]CreateInferInput failed, ret %d", ret);
            return;
        }

        ret = inferEngine->CreateInferOutput();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]CreateInferOutput failed, ret %d", ret);
            inferEngine->DestroyDataset(inferEngine->input_);
            return;
        }

        ret = inferEngine->ExecInference();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]AsynInferenceExecute failed");
            inferEngine->DestroyDataset(inferEngine->input_);
            inferEngine->DestroyDataset(inferEngine->output_);
            return;
        }

        int ret32 = inferEngine->SaveInferResult(inferEngine->output_, &inferFile_vec);
        if (ret != 0) {
            LOG_ERROR("[ERROR]SaveInferResult failed, ret %d", ret32);
        }

        inferEngine->DestroyDataset(inferEngine->input_);
        inferEngine->DestroyDataset(inferEngine->output_);
    }
}

aclError multi_inputs_Inference_engine::Multi_Inputs_InferenceThreadProc()
{
    std::thread inferThread(Multi_Inputs_InferThread, this);
    threads_.push_back(std::move(inferThread));
    return ACL_ERROR_NONE;
}

multi_inputs_Inference_engine::~multi_inputs_Inference_engine()
{
    LOG_INFO("multi_inputs_Inference_engine deconstruct.");
}
