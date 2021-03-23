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

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <vector>
#include <sys/time.h>

#include "face_feature.h"
#include "utils.h"
#include "save_feature.h"
#include "normalization.h"

using namespace std;

namespace {
    uint32_t featureModelWidth = 112;
    uint32_t featureModelHeight = 112;
    const char* kModelPath = "./model/face_feature.om";
}

int main(int argc, char *argv[]) {
    //Check the input when the application is executed, the program execution requires the input of picture directory parameters
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }
    //Instantiate the target detection object, the parameter is the classification model path, the width and height required by the model input
    FaceFeature feature(kModelPath,featureModelWidth, featureModelHeight);
    //Initialize the acl resources, models and memory for classification inference
    Result ret = feature.init();
    if (ret != SUCCESS) {
        ERROR_LOG("Classification Init resource failed");
        return FAILED;
    }

    //Get all the image file names in the image directory
    string inputImageDir = string(argv[1]);
    vector<string> fileVec;
    Utils::get_all_files(inputImageDir, fileVec);
    if (fileVec.empty()) {
        ERROR_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }

    //Reasoning picture by picture
    for (string imageFile : fileVec) {
        //Utils::ReadImageFile(image, imageFile);
        ImageData rgbImage;
        Result ret = feature.pre_process(imageFile, rgbImage);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
            imageFile.c_str());
            continue;
        }
        aclmdlDataset* inferenceOutput = nullptr;

        struct timeval startTime, endTime;
        gettimeofday(&startTime, NULL);

        for(int i = 0; i < 1000; i++) {
            ret = feature.inference(inferenceOutput, rgbImage);
            if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
        }
        gettimeofday(&endTime, NULL);
        long time = (endTime.tv_sec * 1000 + endTime.tv_usec / 1000) - (startTime.tv_sec * 1000 + startTime.tv_usec / 1000);
        INFO_LOG("Inference average time without first time: %ld", time / 1000);


        FaceInfo faceInfo;
        int pos = imageFile.find_last_of("/");
        int end = imageFile.find_last_of(".");
        string fileName = imageFile.substr(pos + 1, (end - pos - 1));
        strcpy(faceInfo.fileName, fileName.c_str());

        //Analyze the inference output and mark the object category and location obtained by the inference on the picture
        ret = feature.post_process(inferenceOutput,faceInfo);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }

        normalization(faceInfo.feature, faceInfo.featureSize);
        save(faceInfo);
    }
    INFO_LOG("Execute sample success");
    return SUCCESS;
}
