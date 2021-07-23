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
/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2099. All Rights Reserved.
 * Description: MIAN
 * Author: Atlas
 * Create: 2020-02-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * ============================================================================
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2 Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3 Neither the names of the copyright holders nor the names of the
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/time.h>
#include "unistd.h"
#include "acl/acl.h"
#include "common/block_queue.h"
#include "common/data_struct.h"
//#include "common/command_line.h"
#include "data_input/data_input.h"
#include "inference/inference.h"
#include "preprocess/preprocess.h"
#include "postprocess/postprocess.h"

aclrtContext context;

/*
bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    std::vector<std::string> validModelType = { "vision", "nlp", "fasterrcnn", "nmt", "widedeep" };
    if (std::find(validModelType.begin(), validModelType.end(), FLAGS_model_type) == validModelType.end()) {
        std::cout << "invalid model type, please check!" << std::endl;
        return false;
    }

    if (FLAGS_batch_size < 1) {
        std::cout << "invalid batch size, please check!" << std::endl;
        return false;
    }

    if (FLAGS_device_id < 0) {
        std::cout << "invalid device id, please check!" << std::endl;
        return false;
    }

    if (FLAGS_input_width < 0) {
        std::cout << "invalid input_width, please check!" << std::endl;
        return false;
    }

    if (FLAGS_input_height < 0) {
        std::cout << "invalid input_width, please check!" << std::endl;
        return false;
    }

    return true;
}
*/
int main(int argc, char **argv)
{
    std::string FLAGS_model_type = string(argv[1]);
    int FLAGS_batch_size = atoi(argv[2]);
    int FLAGS_device_id = atoi(argv[3]);
    std::string FLAGS_om_path = string(argv[4]);
    std::string FLAGS_input_text_path = string(argv[5]);
    // noused, for other network
    int FLAGS_input_width = 224;
    int FLAGS_input_height = 224;
    std::string FLAGS_input_vocab = "";
    bool FLAGS_output_binary = false;
    std::string FLAGS_ref_vocab = "";

    struct timeval e2eStart, e2eEnd;
    gettimeofday(&e2eStart, nullptr);

    /*
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }
    */
    ModelType mtype = MT_WIDEDEEP;
    if (FLAGS_model_type == "vision") {
        mtype = MT_VISION;
    } else if (FLAGS_model_type == "nmt") {
        mtype = MT_NMT;
    } else if (FLAGS_model_type == "widedeep") {
        mtype = MT_WIDEDEEP;
    } else if (FLAGS_model_type == "nlp") {
        mtype = MT_NLP;
    } else if (FLAGS_model_type == "fasterrcnn") {
        mtype = MT_FASTERRCNN;
    }

    aclInit("");
    aclrtSetDevice(FLAGS_device_id);
    aclrtCreateContext(&context, FLAGS_device_id);

    /* define the queues between different modules */
    BlockingQueue<std::shared_ptr<RawData>> *inputToPreQueue = new BlockingQueue<std::shared_ptr<RawData>>;
    BlockingQueue<std::shared_ptr<RawData>> *inputToPreQueue1 = new BlockingQueue<std::shared_ptr<RawData>>;
    BlockingQueue<std::shared_ptr<ModelInputData>> *preToInferQueue =
        new BlockingQueue<std::shared_ptr<ModelInputData>>;
    BlockingQueue<std::shared_ptr<ModelOutputData>> *inferToPostQueue =
        new BlockingQueue<std::shared_ptr<ModelOutputData>>;
    BlockingQueue<std::shared_ptr<ModelOutputData>> *preToPostQueue =
        new BlockingQueue<std::shared_ptr<ModelOutputData>>;

    /* create modules and init */
    Preprocess *preprocessInstance = new Preprocess;
    DataManager *dataManagerInstance = new DataManager;
    InferBase *inferenceInstance = new InferBase(FLAGS_batch_size, FLAGS_device_id);
    PostProcess *postProcessInstance = new PostProcess;


    int32_t status = 0;

    if (mtype == MT_VISION || mtype == MT_FASTERRCNN) {
        status = preprocessInstance->Init(mtype, FLAGS_input_width, FLAGS_input_height);
    } else if (mtype == MT_NMT || mtype == MT_NLP || mtype == MT_WIDEDEEP) {
        status = preprocessInstance->Init(mtype, FLAGS_input_vocab);
    }

    if (!status) {
        std::cout << "[ERROR] preprocessInstance init failed." << std::endl;
        return false;
    }
    dataManagerInstance->Init(mtype, FLAGS_input_text_path);
    status = inferenceInstance->Init(FLAGS_om_path);
    if (!status) {
        std::cout << "[ERROR] inferenceInstance init failed." << std::endl;
        return false;
    }

    if (mtype == MT_NMT) {
        status = postProcessInstance->Init(FLAGS_batch_size, mtype, FLAGS_ref_vocab, FLAGS_output_binary);
        if (!status) {
            std::cout << "[ERROR] postProcessInstance init failed." << std::endl;
            return false;
        }
    } else {
        status = postProcessInstance->Init(FLAGS_batch_size, mtype, FLAGS_input_text_path, FLAGS_output_binary);
        if (!status) {
            std::cout << "[ERROR] postProcessInstance init failed." << std::endl;
            return false;
        }
    }

    /* run the modules */
    if (mtype == MT_WIDEDEEP) {
        preprocessInstance->Run(inputToPreQueue, inputToPreQueue1, preToInferQueue, preToPostQueue);
        dataManagerInstance->Run(inputToPreQueue, inputToPreQueue1);
        postProcessInstance->Run(inferToPostQueue, preToPostQueue);
    } else {
        preprocessInstance->Run(inputToPreQueue, nullptr, preToInferQueue, nullptr);
        dataManagerInstance->Run(inputToPreQueue);
        postProcessInstance->Run(inferToPostQueue, nullptr);
    }
    inferenceInstance->Run(preToInferQueue, inferToPostQueue);

    /* wait */
    const int sleepTime = 0.5;
    while (!postProcessInstance->GetFinishFlag()) {
        sleep(sleepTime);
    }

    gettimeofday(&e2eEnd, nullptr);
    const float mulTimes = 1000.0;
    double e2eCost = (e2eEnd.tv_sec - e2eStart.tv_sec) * mulTimes + (e2eEnd.tv_usec - e2eStart.tv_usec) / mulTimes;
    /* print and save the perf info */
    std::vector<std::shared_ptr<PerfInfo>> perfInfoPre = preprocessInstance->GetPerfInfo();
    std::shared_ptr<PerfInfo> perfInfoData = dataManagerInstance->GetPerfInfo();
    std::shared_ptr<PerfInfo> perfInfoInfer = inferenceInstance->GetPerfInfo();
    std::shared_ptr<PerfInfo> perfInfoPost = postProcessInstance->GetPerfInfo();


    std::string outFileName = "perf_" + FLAGS_model_type + "_batchsize_" + std::to_string(FLAGS_batch_size) +
        "_device_" + std::to_string(FLAGS_device_id) + ".txt";
    std::ofstream outFile(outFileName);
    std::cout << "-----------------Performance Summary------------------" << std::endl;
    if (mtype == MT_WIDEDEEP) {
        std::cout << "[e2e] throughputRate: " << (perfInfoData->count / e2eCost) * mulTimes << ", lantency: " << e2eCost
                  << std::endl
                  << "[data read] throughputRate: " << perfInfoData->throughputRate
                  << ", moduleLantency: " << perfInfoData->moduleLantency << std::endl
                  << "[wide preprocess] throughputRate: " << perfInfoPre[0]->throughputRate
                  << ", moduleLantency: " << perfInfoPre[0]->moduleLantency << std::endl
                  << "[deep preprocess] throughputRate: " << perfInfoPre[1]->throughputRate
                  << ", moduleLantency: " << perfInfoPre[1]->moduleLantency << std::endl
                  << "[infer] throughputRate: " << perfInfoInfer->throughputRate
                  << ", Interface throughputRate: " << (1.0 / perfInfoInfer->inferLantency * mulTimes)
                  << ", moduleLantency: " << perfInfoInfer->moduleLantency << std::endl
                  << "[post] throughputRate: " << perfInfoPost->throughputRate
                  << ", moduleLantency: " << perfInfoPost->moduleLantency << std::endl;

        outFile << "[e2e] throughputRate: " << (perfInfoData->count / e2eCost) * mulTimes << ", lantency: " << e2eCost
                << std::endl
                << "[data read] throughputRate: " << perfInfoData->throughputRate
                << ", moduleLantency: " << perfInfoData->moduleLantency << std::endl
                << "[wide preprocess] throughputRate: " << perfInfoPre[0]->throughputRate
                << ", moduleLantency: " << perfInfoPre[0]->moduleLantency << std::endl
                << "[deep preprocess] throughputRate: " << perfInfoPre[1]->throughputRate
                << ", moduleLantency: " << perfInfoPre[1]->moduleLantency << std::endl
                << "[infer] throughputRate: " << perfInfoInfer->throughputRate
                << ", Interface throughputRate: " << (1.0 / perfInfoInfer->inferLantency * mulTimes)
                << ", moduleLantency: " << perfInfoInfer->moduleLantency << std::endl
                << "[post] throughputRate: " << perfInfoPost->throughputRate
                << ", moduleLantency: " << perfInfoPost->moduleLantency << std::endl;
    } else {
        std::cout << "[e2e] throughputRate: " << (perfInfoData->count / e2eCost) * mulTimes << ", lantency: " << e2eCost
                  << std::endl
                  << "[data read] throughputRate: " << perfInfoData->throughputRate
                  << ", moduleLantency: " << perfInfoData->moduleLantency << std::endl
                  << "[preprocess] throughputRate: " << perfInfoPre[0]->throughputRate
                  << ", moduleLantency: " << perfInfoPre[0]->moduleLantency << std::endl
                  << "[infer] throughputRate: " << perfInfoInfer->throughputRate
                  << ", Interface throughputRate: " << (1.0 / perfInfoInfer->inferLantency * mulTimes)
                  << ", moduleLantency: " << perfInfoInfer->moduleLantency << std::endl
                  << "[post] throughputRate: " << perfInfoPost->throughputRate
                  << ", moduleLantency: " << perfInfoPost->moduleLantency << std::endl;

        outFile << "[e2e] throughputRate: " << (perfInfoData->count / e2eCost) * mulTimes << ", lantency: " << e2eCost
                << std::endl
                << "[data read] throughputRate: " << perfInfoData->throughputRate
                << ", moduleLantency: " << perfInfoData->moduleLantency << std::endl
                << "[preprocess] throughputRate: " << perfInfoPre[0]->throughputRate
                << ", moduleLantency: " << perfInfoPre[0]->moduleLantency << std::endl
                << "[infer] throughputRate: " << perfInfoInfer->throughputRate
                << ", Interface throughputRate: " << (1.0 / perfInfoInfer->inferLantency * mulTimes)
                << ", moduleLantency: " << perfInfoInfer->moduleLantency << std::endl
                << "[post] throughputRate: " << perfInfoPost->throughputRate
                << ", moduleLantency: " << perfInfoPost->moduleLantency << std::endl;
    }
    outFile.close();
    std::cout << "-----------------------------------------------------------" << std::endl;

    /* finish the modules */
    preprocessInstance->DeInit();
    inferenceInstance->UnInit();
    dataManagerInstance->DeInit();
    postProcessInstance->DeInit();

    delete preprocessInstance;
    delete dataManagerInstance;
    delete inferenceInstance;
    delete postProcessInstance;

    delete inputToPreQueue;
    delete inputToPreQueue1;
    delete preToInferQueue;
    delete inferToPostQueue;
    delete preToPostQueue;

    aclrtDestroyContext(context);
    aclrtResetDevice(FLAGS_device_id);
    return 0;
}
