/*
* @file utils.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
* Description: model_process
* Author: fuyangchenghu
* Create: 2020/6/22
* Notes:
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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
};

#endif
