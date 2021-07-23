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

#include <limits.h>
#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "data_input.h"

void DataManager::Init(ModelType type, const string &path)
{
    modelType_ = type;
    configPath_ = path;
    cout << "[INFO][DataManager] Init SUCCESS" << endl;
}

void DataManager::DeInit()
{
    alive_ = false;
    if (getDataInfoThread_.joinable()) {
        getDataInfoThread_.join();
    }
    cout << "[INFO][DataManager] DeInit SUCCESS" << endl;
}

void DataManager::Run(BlockOutputQueue *outQueue, BlockOutputQueue *outQueue2)
{
    alive_ = true;
    outputQueue_ = outQueue;
    wideDeepQueue_ = outQueue2;
    perfInfo_.reset(new PerfInfo);
    getDataInfoThread_ = thread(&DataManager::GetDataInfo, this);
    return;
}

void DataManager::GetDataInfo()
{
    gettimeofday(&managerStart_, NULL);
    char actualPath[PATH_MAX +1] = {0x00};
    if (configPath_.length() > PATH_MAX || realpath(configPath_.c_str(), actualPath) == NULL) {
        cout << "[ERROR][DataManager] Get absolute path failed!" << endl;
        return;
    }

    ifstream fin(actualPath);
    if (!fin.is_open()) {
        cout << "[ERROR][DataManager] Open file failed!" << endl;
        return;
    }

    if (modelType_ == MT_VISION || modelType_ == MT_FASTERRCNN) {
        GetImageInfo(fin);
    } else if (modelType_ == MT_NLP || modelType_ == MT_NMT || modelType_ == MT_WIDEDEEP) {
        GetTextInfo(fin);
    } else {
        cout << "[ERROR][DataManager] Unsupport model type." << endl;
    }

    fin.close();
}

bool DataManager::CheckEOF(ifstream &fin, string &content)
{
    content.clear();
    bool isEOF = false;
    while (content.empty() && !isEOF) {
        isEOF = (getline(fin, content)) ? false : true;
    }
    return isEOF;
}

void DataManager::CalculatePerTime(uint32_t id)
{
    struct timeval end;
    gettimeofday(&end, NULL);
    const float usecUnit = 1000000.0;
    const float msecUnit = 1000.0;
    double curTimeCost_ = (end.tv_sec - managerStart_.tv_sec) + (end.tv_usec - managerStart_.tv_usec) / usecUnit;
    perfInfo_->throughputRate = id / curTimeCost_;
    if (id != 0) {
        perfInfo_->moduleLantency = (curTimeCost_ / id) * msecUnit; // ms
    } else {
        cout << "[ERROR][DataManager] id == 0." << endl;
    }

    perfInfo_->count = id; // sample count
}

void DataManager::GetTextInfo(ifstream &fin)
{
    uint32_t id = 0;
    string content;
    bool isEOF = CheckEOF(fin, content);
    while (alive_ && !isEOF) {
        if (modelType_ == MT_WIDEDEEP) {
            replace(content.begin(), content.end(), ',', ' ');
        }
        shared_ptr<RawData> output = make_shared<RawData>();
        output->dataId = id++;
        output->modelType = modelType_;
        stringstream lineStr(content);
        while (true) {
            string attr;
            lineStr >> attr;
            if (lineStr.fail()) {
                break;
            }
            DataBuf data;
            data.len = attr.length() + 1;
            char *info = new char[data.len];
            attr.copy(info, data.len, 0);
            info[data.len - 1] = 0;
            data.buf.reset((uint8_t *)info, [](uint8_t *p) { delete[] p; });
            output->text.textRawData.push_back(data);
        }
        isEOF = CheckEOF(fin, content);
        output->finish = isEOF;
        outputQueue_->Push(output);
        if (modelType_ == MT_WIDEDEEP) {
            wideDeepQueue_->Push(output);
        }
        CalculatePerTime(id);
    }
}

void DataManager::GetImageInfo(ifstream &fin)
{
    string content;
    bool isEOF = CheckEOF(fin, content);
    uint32_t id = 0;
    while (alive_ && !isEOF) {
        ImageInfo info;
        stringstream lineStr(content);
        lineStr >> info.id >> info.path >> info.width >> info.height;

        char fullPath[PATH_MAX +1] = {0x00};
        if (info.path.length() > PATH_MAX || realpath(info.path.c_str(), fullPath) == NULL) {
            cout << "[ERROR][DataManager] Get absolute path failed!" << endl;
            continue;
        }

        uint32_t bufSize = 0;
        char *bufData = ReadBinFile(fullPath, bufSize);
        if (bufData == nullptr || bufSize == 0) {
            continue;
        }
        shared_ptr<RawData> output = make_shared<RawData>();
        output->dataId = info.id;
        output->modelType = modelType_;
        output->img.width = info.width;
        output->img.height = info.height;
        output->img.data.buf.reset((uint8_t *)bufData, [](uint8_t *p) { delete[] p; });
        output->img.data.len = bufSize;
        isEOF = CheckEOF(fin, content);
        output->finish = isEOF;
        outputQueue_->Push(output);
        CalculatePerTime(++id);
    }
}

char *DataManager::ReadBinFile(const char *fileName, uint32_t &fileSize)
{
    filebuf *pbuf;
    ifstream fileStream;
    size_t size;
    fileStream.open(fileName, ios::binary);
    if (!fileStream) {
        cout << "[ERROR][DataManager] open jpeg file failed: " << fileName << endl;
        return nullptr;
    }

    pbuf = fileStream.rdbuf();
    size = pbuf->pubseekoff(0, ios::end, ios::in);
    pbuf->pubseekpos(0, ios::in);
    char *buffer = nullptr;
    buffer = new char[size];
    if (buffer == nullptr) {
        cout << "[ERROR][DataManager] Malloc host buff failed." << endl;
        return nullptr;
    }

    pbuf->sgetn(buffer, size);
    fileSize = size;
    fileStream.close();
    return buffer;
}

shared_ptr<PerfInfo> DataManager::GetPerfInfo()
{
    return perfInfo_;
}
