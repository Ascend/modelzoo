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
#ifndef __HW_LOG_H__
#define __HW_LOG_H__
#include <stdarg.h>

// log level
typedef enum LogLevel {
    LOG_FATAL_LEVEL = 0,
    LOG_ERROR_LEVEL = 1,
    LOG_WARNING_LEVEL = 2,
    LOG_INFO_LEVEL = 3,
    LOG_DEBUG_LEVEL = 4
} LogLevel;

typedef void (*HW_CALLBACK_LOG_FXN)(int level, const char *fmt, va_list arglist);

extern HW_CALLBACK_LOG_FXN g_LogFxn;
extern LogLevel g_ELogLevel;

const char g_log_level[5][20] = {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"};

#define LOG_MODULE_SUB(module, outlevel, level, modulename, fmt, ...)                                                 \
    LogMsg(module, outlevel, level, "\t%-8s  %-22.22s %-4d %-31.31s " fmt, modulename, HW_GetFileShortName(__FILE__), \
        (const char *)__LINE__, (const char *)__FUNCTION__, ##__VA_ARGS__)
#define LOG_SUB(level, modulename, fmt, ...) \
    LOG_MODULE_SUB(g_LogFxn, g_ELogLevel, level, modulename, fmt, ##__VA_ARGS__)
#define HW_LOG(level, fmt, ...) LOG_SUB(level, g_log_level[level], fmt, ##__VA_ARGS__)

#define LOG_DEBUG(fmt, ...) HW_LOG(LOG_DEBUG_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) HW_LOG(LOG_WARNING_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) HW_LOG(LOG_INFO_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) HW_LOG(LOG_ERROR_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) HW_LOG(LOG_FATAL_LEVEL, fmt, ##__VA_ARGS__)

#define RETURN_VAL_IF_FAIL(expr, ret, msgfmt, args...) do {                         \
        if (!(expr)) {                                 \
            LOG_ERROR(msgfmt, ##args);                 \
            return ret;                                \
        }                                              \
    } while (0) 
    
#define RETURN_IF_FAIL(expr, msgfmt, args...) do {                \
        if (!(expr)) {                        \
            LOG_ERROR(msgfmt, ##args);        \
            return;                           \
        }                                     \
    } while (0)


void HW_FuncNotUse(int level, const char *fmt, va_list arglist);
const char *HW_GetFileShortName(const char *fileName);
int LogMsg(HW_CALLBACK_LOG_FXN g_LogFxn, LogLevel outLevel, LogLevel level, const char *fmt, ...);

#endif
