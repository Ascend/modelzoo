/* 
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2099. All Rights Reserved.
 * Description: 数据读取
 * Author: Atlas
 * Create: 2020-02-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * ============================================================================
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


#ifndef BENCHMARK_DATA_INPUT_H
#define BENCHMARK_DATA_INPUT_H

#include <string>
#include <thread>
#include "common/block_queue.h"
#include "common/data_struct.h"

using namespace std;

struct ImageInfo {
    uint32_t id;
    string path;
    uint32_t width;
    uint32_t height;
};

class DataManager {
public:
    using BlockOutputQueue = BlockingQueue<shared_ptr<RawData>>;
    void Init(ModelType type, const string &path);
    void DeInit();
    void Run(BlockOutputQueue *outQueue, BlockOutputQueue *outQueue1 = nullptr);
    shared_ptr<PerfInfo> GetPerfInfo();

private:
    void GetDataInfo();
    void GetImageInfo(ifstream &fin);
    void GetTextInfo(ifstream &fin);
    char *ReadBinFile(const char *fileName, uint32_t &fileSize);
    bool CheckEOF(ifstream &fin, string &content);
    void CalculatePerTime(uint32_t id);

    ModelType modelType_;
    string configPath_;
    BlockOutputQueue *outputQueue_ = nullptr;
    BlockOutputQueue *wideDeepQueue_ = nullptr;
    bool alive_ = false;
    thread getDataInfoThread_;

    // performce
    shared_ptr<PerfInfo> perfInfo_;
    struct timeval managerStart_;
};
#endif // BENCHMARK_DATA_INPUT_H
