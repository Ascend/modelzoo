/* *
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common.h"
#include "hw_log.h"
#include <fstream>
#include <nlohmann/json.hpp>

#include "ptest.h"
#include "json.h"
#include <unistd.h>
#include <map>

using json = nlohmann::json;

#define TEST_CASE_SINGLE_EXEC 2
#define TEST_CASE_CYCLIE_EXEC 3
using namespace std;
char g_logFileName[256];

int init_env()
{
    printf("init_env START!!!\n");
    printf("init_env SUCCESS!!!\n");
    return 0;
}

int run_case(char *script, char *case_name)
{
    CASE_START(case_name, script);
    ptest::TestResult result = ptest::TestManager::GetSingleton().RunTest(case_name);

    if (ptest::NOCASEFOUND == result) {
        LOG_ERROR("Cannot find testcase: %s", case_name);
    } else {
        std::string message;
        switch (result) {
            case ptest::SUCCESS:
                message = "SUCCESS";
                break;
            case ptest::FAILED:
                message = "FAIL";
                break;
            case ptest::UNAVAILABLE:
                message = "UNAVAILABLE";
                break;
            default:
                message = "UNKNOWN";
                break;
        }
        printf("%s %s\n", case_name, message.c_str());
        LOG_INFO("%s %s", case_name, message.c_str());
    }

    CASE_END(case_name);
    return (int)result;
}

int isStartsWith(const char *str1, char *str2)
{
    if (str1 == NULL || str2 == NULL)
        return -1;
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    if ((len1 < len2) || (len1 == 0 || len2 == 0))
        return -1;
    char *p = str2;
    int i = 0;
    while (*p != '\0') {
        if (*p != str1[i])
            return 0;
        p++;
        i++;
    }
    return 1;
}

int DemoLogInit(LogLevel level, HW_CALLBACK_LOG_FXN fxn)
{
    if (!fxn) {
        return -1;
    }
    if (level > LOG_DEBUG_LEVEL) {
        return -1;
    }
    g_LogFxn = (HW_CALLBACK_LOG_FXN)fxn;
    g_ELogLevel = (LogLevel)level;
    LOG_INFO("%s", "Log init success");
    return 0;
}

void ImrsLogCallback(int level, const char *fmt, va_list vl)
{
    char buf[1024];
    struct timeval tv;
    struct tm *p;
    FILE *fp;
    fp = fopen(g_logFileName, "a+");
    if (fp) {
        gettimeofday(&tv, NULL);
        p = localtime(&tv.tv_sec);
        fprintf(fp, "%d-%d-%d: %d:%d:%d.%ld  ", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min,
            p->tm_sec, tv.tv_usec);
        vsnprintf(buf, sizeof(buf), fmt, vl);
        fprintf(fp, "%s\n", buf);
        fclose(fp);
    }
}

int main(int argc, char **argv)
{
    int ret = 0;

    sprintf(g_logFileName, "./ACL_testcase.log");
    DemoLogInit(LOG_INFO_LEVEL, ImrsLogCallback);

    if (argc <= 1 || argc > 3) {
        LOG_ERROR("wrong arguments!!!\n");
        return -1;
    }

    ret = getConfigFromJsonFile(argv[1]);
    if (ret != 0) {
        LOG_ERROR("getConfigFromJsonFile fail!!!\n");
        return -1;
    }

    if (argc == TEST_CASE_CYCLIE_EXEC) {
        return 0;
    } else if (argc == TEST_CASE_SINGLE_EXEC) {
        run_case(argv[0], const_cast<char *>(testcaseName.c_str()));
    }

    return 0;
}
