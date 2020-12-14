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

#include <bits/types/struct_timeval.h>
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <sys/time.h>

#include "crowd_count.h"
#include "utils.h"

namespace {
uint32_t kModelWidth = 1408;
uint32_t kModelHeight = 800;
const char* kModelPath = "./model/count_person.caffe.om";
}

int main(int argc, char *argv[]) {
    //Check the input when the application is executed, the program execution requires the input of picture directory parameters
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }
    //Instantiate the target detection object, the parameter is the classification model path, the width and height required by the model input
    CrowdCount count(kModelPath, kModelWidth, kModelHeight);
    //Initialize the acl resources, models and memory for classification inference
    Result ret = count.init();
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
    ImageData image;
    for (string imageFile : fileVec) {
        Utils::read_image_file(image, imageFile);
        if (image.data == nullptr) {
            ERROR_LOG("Read image %s failed", imageFile.c_str());
            return FAILED;
        }

        //Preprocess the picture: read the picture and zoom the picture to the size required by the model input
        ImageData resizedImage;
        //Result ret = detect.Preprocess(imageFile, resizedImage);
        Result ret = count.pre_process(resizedImage,image);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
                      imageFile.c_str());                
            continue;
        }

        //Send the preprocessed pictures to the model for inference and get the inference results
        aclmdlDataset* inferenceOutput = nullptr;

        struct timeval start, end;

        gettimeofday(&start, NULL);

        for(int i = 0; i < 1000; i++) {
            ret = count.inference(inferenceOutput, resizedImage);
            if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
        }
        gettimeofday(&end, NULL);
        long time = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
        INFO_LOG("Inference average time without first time: %ld", time / 1000);
        //Analyze the inference output and mark the object category and location obtained by the inference on the picture
        ret = count.post_process(resizedImage, inferenceOutput, imageFile);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
    }

    INFO_LOG("Execute sample success");
    return SUCCESS;
}
