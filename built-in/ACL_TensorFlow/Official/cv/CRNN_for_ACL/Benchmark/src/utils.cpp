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

#include "utils.h"
#include "acl/acl.h"
#include <sys/time.h>
using namespace std;
extern bool g_isDevice;
extern bool f_isTXT;
extern int32_t device;
extern bool is_profi;
extern bool is_dump;

void* Utils::ReadBinFile(std::string fileName, uint32_t& fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        cout<<"qwe"<<endl;
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);

    void* binFileBufferData = nullptr;
    aclError ret = ACL_ERROR_NONE;
    if (!g_isDevice) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (binFileBufferData == nullptr) {
            ERROR_LOG("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    }

    binFile.read(static_cast<char*>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

void* Utils::GetDeviceBufferOfFile(std::string fileName, uint32_t& fileSize)
{
    uint32_t inputHostBuffSize = 0;
    void* inputHostBuff = Utils::ReadBinFile(fileName, inputHostBuffSize);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }
    if (!g_isDevice) {
        void* inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        aclError ret = aclrtMalloc(&inBufferDev, inBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                inBufferSize, inputHostBuffSize);
            aclrtFree(inBufferDev);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }
        aclrtFreeHost(inputHostBuff);
        fileSize = inBufferSize;
        return inBufferDev;
    } else {
        fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}

void Utils::SplitString(std::string& s, std::vector<std::string>& v, char c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        std::string s1 = s.substr(pos1, pos2 - pos1);
        size_t n = s1.find_last_not_of(" \r\n\t");
        if (n != string::npos) {
            s1.erase(n + 1, s.size() - n);
        }
        n = s1.find_first_not_of(" \r\n\t");
        if (n != string::npos) {
            s1.erase(0, n);
        }
        v.push_back(s1);
        pos1 = pos2 + 1;
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) {
        std::string s1 = s.substr(pos1);
        size_t n = s1.find_last_not_of(" \r\n\t");
        if (n != string::npos) {
            s1.erase(n + 1, s.size() - n);
        }
        n = s1.find_first_not_of(" \r\n\t");
        if (n != string::npos) {
            s1.erase(0, n);
        }
        v.push_back(s1);
    }
}

int Utils::str2num(char* str)
{
    int n = 0;
    int flag = 0;
    while (*str >= '0' && *str <= '9') {
        n = n * 10 + (*str - '0');
        str++;
    }
    if (flag == 1) {
        n = -n;
    }
    return n;
}

std::string Utils::modelName(string& s)
{
    string::size_type position1, position2;
    position1 = s.find_last_of("/");
    if (position1 == s.npos) {
        position1 = 0;
    }else{position1 = position1 + 1;
    }
    position2 = s.find_last_of(".");
    std::string modelName = s.substr(position1, position2 - position1);
    return modelName;
}

std::string Utils::TimeLine()
{
    time_t currentTime = time(NULL);
    char chCurrentTime[64];
    strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d_%H%M%S", localtime(&currentTime));
    std::string stCurrentTime = chCurrentTime;
    return stCurrentTime;
}

void Utils::printCurrentTime()
{
    char szBuf[256] = { 0 };
    struct timeval tv;
    struct timezone tz;
    struct tm* p = nullptr;

    gettimeofday(&tv, &tz);
    p = localtime(&tv.tv_sec);
    printf("%02d-%02d-%02d %02d:%02d:%02d.%06ld\n", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec);
}
void Utils::printHelpLetter()
{
    cout << endl;
    cout << "Usage:" << endl;
    cout << "generate offline model inference output file example:" << endl;
    cout << "./msame --model /home/HwHiAiUser/ljj/colorization.om --input /home/HwHiAiUser/ljj/colorization_input.bin --output /home/HwHiAiUser/ljj/AMEXEC/out/output1 --outfmt TXT --loop 2" << endl
         << endl;

    cout << "arguments explain:" << endl;
    cout << "  --model       Model file path" << endl;
    cout << "  --input	Input data path(only accept binary data file) 	If there are several file, please seprate by ','" << endl;
    cout << "  --output	Output path(User needs to have permission to create directories)" << endl;
    cout << "  --outfmt	Output file format (TXT or BIN)" << endl;
    cout << "  --loop 	loop time(must in 1 to 100)" << endl;
    cout << "  --dump	Enable dump (true or false)" << endl;
    cout << "  --profiler	Enable profiler (true or false)" << endl;
    cout << "  --device      Designated the device ID(must in 0 to 255)" << endl;
    cout << "  --debug       Debug switch,print model information (true or false)" << endl;
    cout << "  --dymBatch 	dynamic batch (Do not support now)" << endl
         << endl
         << endl;
}



double Utils::printDiffTime(time_t begin, time_t end)
{
    double diffT = difftime(begin, end);
    printf("The inference time is: %f millisecond\n", 1000 * diffT);
    return diffT * 1000;
}

double Utils::InferenceTimeAverage(double* x, int len)
{
    double sum = 0;
    for (int i = 0; i < len; i++)
        sum += x[i];
    return sum / len;
}

double Utils::InferenceTimeAverageWithoutFirst(double* x, int len)
{
    double sum = 0;
    for (int i = 0; i < len; i++)
        if (i != 0) {
            sum += x[i];
        }

    return sum / (len - 1);
}

void Utils::ProfilerJson(bool isprof, map<char, string>& params)
{
    if (isprof) {
        std::string out_path = params['o'].c_str();
        std::string out_profiler_path = out_path + "/profiler";
        ofstream outstr("acl.json", ios::out);
        outstr << "{\n\"profiler\": {\n    \"switch\": \"on\",\n";
        outstr << "\"output\": \"" << out_profiler_path << "\",\n    ";
        outstr << "\"aic_metrics\": \"\"}\n}";
        outstr.close();

        //mkdir profiler output dir
        const char* temp_s = out_path.c_str();
        if (NULL == opendir(temp_s)) {
            mkdir(temp_s, 0775);
        }
        const char* temp_s1 = out_profiler_path.c_str();
        if (NULL == opendir(temp_s1)) {
            mkdir(temp_s1, 0775);
        }
    }
}

void Utils::DumpJson(bool isdump, map<char, string>& params)
{
    if (is_dump) {
        std::string modelPath = params['m'].c_str();
        std::string modelName = Utils::modelName(modelPath);
        std::string out_path = params['o'].c_str();
        std::string out_dump_path = out_path + "/dump";
        ofstream outstr("acl.json", ios::out);
        outstr << "{\n\"dump\": {\n    \"dump_path\": \"";
        outstr << out_dump_path << "\",\n    ";
        outstr << "\"dump_mode\": \"output\",\n    \"dump_list\": [{\n    ";
        outstr << "        \"model_name\": \"" << modelName << "\"\n        }]\n";
        outstr << "    }\n}";
        outstr.close();

        //mkdir dump output dir
        const char* temp_s = out_path.c_str();
        if (NULL == opendir(temp_s)) {
            mkdir(temp_s, 0775);
        }
        const char* temp_s1 = out_dump_path.c_str();
        if (NULL == opendir(temp_s1)) {
            mkdir(temp_s1, 0775);
        }
    }
}

int Utils::ScanFiles(std::vector<std::string> &fileList, std::string inputDirectory)
{
    const char* str= inputDirectory.c_str();
    DIR* dir= opendir(str);
    struct dirent* p= NULL;
    while((p= readdir(dir)) != NULL )
    {
        if (p->d_name[0] != '.')
        {
            string name = string(p->d_name);
            fileList.push_back(name);
        }
    }
    closedir(dir);
    if (fileList.size() ==0)
    {
        printf("[ERROR] No file in the directory[%s]", str);
    }
    return fileList.size();
}
