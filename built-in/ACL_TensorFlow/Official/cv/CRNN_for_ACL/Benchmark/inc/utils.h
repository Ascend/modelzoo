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

#ifndef _UTILS_H_
#define _UTILS_H_
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <vector>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define DEBUG_LOG(fmt, args...) fprintf(stdout, "[DEBUG] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

//static size_t loop = 1;
typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

/**
* Utils
*/
class Utils {
public:
    /**
    * @brief create device buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return device buffer of file
    */
    static void* GetDeviceBufferOfFile(std::string fileName, uint32_t& fileSize);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return buffer of pic
    */
    static void* ReadBinFile(std::string fileName, uint32_t& fileSize);

    static void SplitString(std::string& s, std::vector<std::string>& v, char c);

    static int str2num(char* str);

    static std::string modelName(std::string& s);

    static std::string TimeLine();

    static void printCurrentTime();

    static void printHelpLetter();

    static double printDiffTime(time_t begin, time_t end);

    static double InferenceTimeAverage(double* x, int len);

    static double InferenceTimeAverageWithoutFirst(double* x, int len);

    static void ProfilerJson(bool isprof, std::map<char, std::string>& params);

    static void DumpJson(bool isdump, std::map<char, std::string>& params);

    static int ScanFiles(std::vector<std::string> &fileList, std::string inputDirectory);
};

#endif
