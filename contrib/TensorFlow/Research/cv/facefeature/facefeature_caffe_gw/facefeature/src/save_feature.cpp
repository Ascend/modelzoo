/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "save_feature.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <fstream>

Result save(FaceInfo &faceInfo) {
    std::fstream outputFile("../faceLib/people.bin", std::ios::out|std::ios::binary|std::ios::app);
    if (!outputFile)
    {
        INFO_LOG("Error writeFaceLib opening file. Program aborting.\n");
        return FAILED;
    }
    outputFile.write(reinterpret_cast<char *>(&faceInfo), sizeof(FaceInfo));
    outputFile.close();
    return SUCCESS;
}

Result read_face_lib(std::vector<FaceInfo> &faceInfos) {
    std::fstream inputFile("../faceLib/people.bin", std::ios::in|std::ios::binary);
    if (!inputFile)
    {
        INFO_LOG("Error readFaceLib opening file. Program aborting.\n");
        return FAILED;
    }
    FaceInfo faceInfo;
    inputFile.read(reinterpret_cast<char *>(&faceInfo), sizeof (FaceInfo));
    while (!inputFile.eof())
    {
        faceInfos.emplace_back(faceInfo);
        inputFile.read(reinterpret_cast<char *>(&faceInfo), sizeof(FaceInfo));
    }
    inputFile.close();
    return SUCCESS;
}