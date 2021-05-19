/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "CrnnRecognition.h"
#include "MxBase/Log/Log.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

namespace CONST_CONFIG{
const uint32_t CLASS_NUM = 37;
const uint32_t OBJECT_NUM = 24;
const uint32_t BLANK_INDEX = 36;
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles)
{
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir: " << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr == readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        imgFiles.emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

void SetInitParam(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.classNum = CONST_CONFIG::CLASS_NUM;
    initParam.objectNum = CONST_CONFIG::OBJECT_NUM;
    initParam.blankIndex = CONST_CONFIG::BLANK_INDEX;
    initParam.labelPath = "../../models/crnn_labels.names";
    initParam.argmax = false;
    initParam.modelPath = "../../models/crnn-YUV400.om"
}

void ShowUsage()
{
    cout << "Usage   : ./crnn <--image or --dir> [Option]" << endl;
    cout << "Options :" << endl;
    cout << " --image infer_image_path    the path of single infer image, such as ./crnn --image /home/infer/images/test.jpg." << endl;
    cout << " --dir infer_image_dir       the dir of batch infer images, such as ./crnn --dir /home/infer/images." << endl;
    return;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "Please use as follows." << endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    std::string option = argv[1];
    std::string imgPath = argv[2];

    if (option != "--image" && option != "--dir") {
        cout << "Please use as follows." << endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    SetInitParam(initParam);

    auto crnn = std::make_shared<CrnnRecognition>();
    APP_ERROR ret = crnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnRecognition init failed, ret=" << ret << ".";
        return ret;
    }

    if (option == "--image"){
        ofstream infer_ret("infer_result_single.txt", ios::app);
        std::string result = "";
        ret = crnn->Process(imgPath, result);
        int file_pos = imgPath.find("//") + 1;
        infer_ret << imgPath.substr(file_pos, -1) << " " << result << endl;
        if (ret != APP_ERR_OK) {
            LogError << "CrnnRecognition process failed, ret=" << ret << ".";
            crnn->DeInit();
            return ret;
        }
        crnn->DeInit();
        infer_ret.close();
        return APP_ERR_OK;
    }

    ofstream infer_ret("infer_result_multi.txt")
    std::vector<std::string> imgFilePaths;
    ScanImages(imgPath, imgFilePaths);
    auto startTime = std::chrono::high_resolution_clock::now();

    for (auto &imgFile : imgFilePaths) {
        std::string result = "";
        ret = crnn->Process(imgFile, result);
        int nPos = imgFile.find("//") + 2;
        std::string fileName = imgFile.substr(nPos, -1);
        infer_ret << fileName << " " << result << endl;
        if (ret != APP_ERR_OK) {
            LogError << "CrnnRecognition process faield, ret=" << ret << ".";
            crnn->DeInit();
            return ret;
        }
    }

    crnn->DeInit();
    auto endTime = std::chrono::high_resolution_clock::now();
    infer_ret.close();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / crnn->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";

    ifstream infer_ret("infer_result_multi.txt")
    int fCount = 0;
    float acc = 0;
    string s;
    while(getline(infer_file, s)){
        int sPos = s.find(" ");
        string fileName = s.substr(0, sPos);
        int sPos1 = fileName.find_last_of("_");
        int sPos2 = fileName.find_last_of(".");
        string label = fileName.substr(sPos1+1, sPos2-sPos1-1);
        string inferRet = s.substr(sPos1+1, -1);

        transform(label.begin(), label.end(), label.begin(), ::tolower);
        if (label == inferRet){
            acc++;
        }
        fCount++;
    }
    infer_file.close();
    cout << "hitted count is " << acc << ", label count is " << fCount << endl;
    cout << "infer accuracy is " << acc/fCount << endl;
    return APP_ERR_OK;
}