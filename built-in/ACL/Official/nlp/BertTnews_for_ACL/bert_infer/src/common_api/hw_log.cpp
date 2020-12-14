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

#include "hw_log.h"
#include "string.h"
#include <stdarg.h>
#include <stdio.h>
using namespace std;
LogLevel g_ELogLevel = LOG_DEBUG_LEVEL;


int LogMsg(HW_CALLBACK_LOG_FXN logFxn, LogLevel outLevel, LogLevel level, const char *fmt, ...)
{
    va_list arglist;
    int ret;
    memset(&arglist, sizeof(va_list), 0);

    if ((level > outLevel) || (NULL == logFxn)) {
        return 0;
    }
    va_start(arglist, fmt);
    logFxn(level, fmt, arglist);
    va_end(arglist);

    return 0;
}


const char *HW_GetFileShortName(const char *fileName)
{
    int shift = 0;
    int i = 0;
    const char *temp = fileName;
    while (*temp != '\0') {
        if ((*temp == '/') || ('\\' == '/')) {
            shift = i;
        }
        i++;
        temp++;
    }
    return fileName + shift + 1;
}


void HW_FuncNotUse(int level, const char *fmt, va_list arglist) {}

HW_CALLBACK_LOG_FXN g_LogFxn = HW_FuncNotUse;
