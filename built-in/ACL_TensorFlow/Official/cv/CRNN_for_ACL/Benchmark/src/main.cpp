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

#include "sample_process.h"
#include "utils.h"
#include <getopt.h>
using namespace std;

bool f_isTXT = false;
bool g_isDevice = false;
int loop = 1;
int32_t device = 0;
bool is_profi = false;
bool is_dump = false;
bool is_debug = false;
string input_Ftype = ".bin";
string model_Ftype = ".om";
string check = "";

void InitAndCheckParams(int argc, char* argv[], map<char, string>& params, vector<string>& inputs)
{
    const char* optstring = "m::i::o::f::hd::p::l::y::e::g::";
    int c, deb, index;
    struct option opts[] = { { "model", required_argument, NULL, 'm' },
        { "input", required_argument, NULL, 'i' },
        { "output", required_argument, NULL, 'o' },
        { "outfmt", required_argument, NULL, 'f' },
        { "help", no_argument, NULL, 1 },
        { "dump", required_argument, NULL, 'd' },
        { "profiler", required_argument, NULL, 'p' },
        { "loop", required_argument, NULL, 'l' },
        { "dymBatch", required_argument, NULL, 'y' },
        { "device", required_argument, NULL, 'e' },
        { "debug", required_argument, NULL, 'g' },
        { 0, 0, 0, 0 } };
    while ((c = getopt_long(argc, argv, optstring, opts, &index)) != -1) {
        switch (c) {
        case 'm':
            check = optarg;
            if (check.find(model_Ftype) != string::npos) {
                params['m'] = optarg;
                break;
            } else {
                printf("input model file type is not .om , please check your model type!\n");
                exit(0);
            }
        case 'i':
            check = optarg;
            params['i'] = optarg;
            Utils::SplitString(params['i'], inputs, ',');
            break;
        case 'o':
            params['o'] = optarg;
            break;
        case 'f':
            params['f'] = optarg;
            break;
        case '?':
            printf("unknown paramenter\n");
            printf("Execute sample failed.\n");
            Utils::printHelpLetter();
            exit(0);
        case 'd':
            params['d'] = optarg;
            break;
        case 'p':
            params['p'] = optarg;
            break;
        case 'l':
            loop = Utils::str2num(optarg);
            cout << "loop:" << loop << endl;
            if (loop > 100 || loop < 1) {
                printf("loop must in 1 to 100\n");
                exit(0);
            }
            break;
        case 'y':
            params['y'] = optarg;
            break;
        case 'e':
            device = Utils::str2num(optarg);
            cout << "device:" << device << endl;
            if (device > 255 || device < 0) {
                printf("device id must in 0 to 255\n");
                exit(0);
            }
            break;
        case 'g':
            params['g'] = optarg;
            break;
        case 1:
            Utils::printHelpLetter();
            exit(0);
        default:
            printf("unknown paramenter\n");
            printf("Execute sample failed.\n");
            Utils::printHelpLetter();
            exit(0);
        }
    }
}

int main(int argc, char* argv[])
{
    map<char, string> params;
    vector<string> inputs;
    InitAndCheckParams(argc, argv, params, inputs);
    printf("******************************\n");
    printf("Test Start!\n");

    if (params.empty()) {
        printf("Invalid params.\n");
        printf("Execute sample failed.\n");
        Utils::printHelpLetter();
        return FAILED;
    }

    if (params['d'].compare("true") == 0) {
        is_dump = true;
    }
    if (params['p'].compare("true") == 0) {
        is_profi = true;
    }
    if (params['g'].compare("true") == 0) {
        is_debug = true;
    }
    if (is_profi && is_dump) {
        ERROR_LOG("dump and profiler can not both be true");
        return FAILED;
    }

    Utils::ProfilerJson(is_profi, params);
    Utils::DumpJson(is_dump, params);

    SampleProcess processSample;

    Result ret = processSample.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Sample init resource failed.");
        return FAILED;
    }

    ret = processSample.Process(params, inputs);
    if (ret != SUCCESS) {
        ERROR_LOG("Sample process failed.");
        return FAILED;
    }

    INFO_LOG("Execute sample success.");
    printf("Test Finish!\n");
    printf("******************************\n");

    return SUCCESS;
}
