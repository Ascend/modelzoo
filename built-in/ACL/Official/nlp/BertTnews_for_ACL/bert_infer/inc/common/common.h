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
#ifndef _COMMON_H_
#define _COMMON_H_

// #include <syslog.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/sem.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/stat.h>
#include <signal.h>
#include <dirent.h>

#include <vector>
#include <cstdio>
#include <string>
#include <iostream>
#include <iomanip>

#include "acl/acl_mdl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#ifndef VERSION_CTRL_CPU_1951
#include "acl/ops/acl_dvpp.h"
#endif
#include "hw_log.h"
#include "ptest.h"

// Macro declaration
#define DEVICE_ID_MAX 64

#define SDK_INFER_MIN(_a, _b) ((_a) > (_b) ? (_b) : (_a))
#define SDK_INFER_MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))

typedef void (*GET_JSON_CONFIG_CALLBACK)(char *jsonFile, void *config);

typedef struct getJsonConfig_s {
    GET_JSON_CONFIG_CALLBACK callback_func;
    void *p_config;
} getJsonConfig_t;

typedef enum COST_MODULE {
    ASYN_MODEL_EXECUTE,
    LAUNCH_CALL_BACK,
    SYN_STREAM,
    POST_PROCESS,

    VDEC_PROCESS,
    VENC_PROCESS,
    JPEGD_PROCESS,
    PNGD_PROCESS,
    VPC_PERF,
    COST_MODULE_NUM
} COST_MODULE;


typedef struct Time_Cost {
    long long perTime[COST_MODULE_NUM][2];
    long long totalTime[COST_MODULE_NUM];
    long long totalCount[COST_MODULE_NUM];
} Time_Cost;

struct DetBox {
    int x1;
    int y1;
    int x2;
    int y2;
    bool isValid;
    DetBox()
    {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
        isValid = false;
    }
};


/* *
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return file data
 */
char *SdkInferReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize);

/* *
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool SdkInferWriteFile(const std::string &filePath, const void *buffer, size_t size);

#ifdef DAVINCI_LITE
#define TEST_LOG(level, format, ...)                                                                          \
    do {                                                                                                      \
        syslog(level, "%lu [ACL]%s:%u " format "\n", syscall(SYS_gettid), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0);

#define LOG2_0_INFO(level, format, ...)                                                                       \
    do {                                                                                                      \
        syslog(level, "%lu [ACL]%s:%u " format "\n", syscall(SYS_gettid), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0);

#else

#define TEST_LOG(level, format, ...)                                                                          \
    do {                                                                                                      \
        syslog(level, "%lu [ACL]%s:%u " format "\n", syscall(SYS_gettid), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0);

#ifndef RUN_ANDROID
typedef union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
} Semun;
#endif
#endif
#define CASE_NAME ptest::TestManager::GetSingleton().GetRunningTestcaseName().c_str()
#define ESL_WAITTIME (30000)
#define FPGA_WAITTIME (30000)
#define TEST_LEVEL 6
#define MAX_UNMATCH_PRINT 10

static aclError threadError = ACL_ERROR_NONE;
static aclError error = ACL_ERROR_NONE;

#define ACL_LOG(format, ...)                                                                                         \
    do {                                                                                                             \
        if (strstr(CASE_NAME, "FUZZ") == NULL) {                                                                     \
            printf("%s:(tid=[%lu],line=[%d])" format "\n", CASE_NAME, syscall(SYS_gettid), __LINE__, ##__VA_ARGS__); \
        }                                                                                                            \
    } while (0);

#define ACL_CLOG(con, format, ...)                                             \
    if ((con))                                                                   \
    {                                                                    \
        printf(format "\n", ##__VA_ARGS__);                              \
    } while (0);

#define ACL_DLOG(format, ...)                                            \
    do                                                                   \
    {                                                                    \
        printf(format "\n", ##__VA_ARGS__);                              \
    } while (0);

#define CASE_START(case_name, script)                     \
    do {                                                  \
        if (strcmp(script, "./benchmark") == 0) {         \
            printf("%s run for online!!!\n", case_name);  \
        } else {                                          \
            printf("%s run for offline!!!\n", case_name); \
        }                                                 \
    } while (0);

#define CASE_END(case_name)                                               \
    do {                                                                  \
        printf("%s END(tid=[%lu])!!!\n", case_name, syscall(SYS_gettid)); \
    } while (0);


inline void checkError(aclError error, aclError e = ACL_ERROR_NONE)
{
    ASSERT_EQ(error, e);
}

inline void checkAssert(aclError error, aclError e = ACL_ERROR_NONE)
{
    printf("%s:(tid=[%lu],line=[%d]) RUN_AST failed,error[%d],expect[%d]\n", CASE_NAME, syscall(SYS_gettid), __LINE__,
        error, e);
    ASSERT_EQ(error, e);
}

inline void checkExpect(aclError error, aclError e = ACL_ERROR_NONE)
{
    EXPECT_EQ(error, e);
}

// xuzhangmin(xwx5322041)add 2020/7/23
char *SdkInferReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize);
// xuzhangmin(xwx5322041)add 2020/7/23
bool SdkInferWriteFile(const std::string &filePath, const void *buffer, size_t size);


#define ALIGN_UP(x, a) ((((x) + ((a)-1)) / a) * a)

#define CHECK_CHN_RET(express, Chn, name)                                                                       \
    do {                                                                                                        \
        HI_S32 Ret;                                                                                             \
        Ret = express;                                                                                          \
        if (HI_SUCCESS != Ret) {                                                                                \
            printf("\033[0;31m%s chn %d failed at %s: LINE: %d with %#x!\033[0;39m\n", name, Chn, __FUNCTION__, \
                __LINE__, Ret);                                                                                 \
            fflush(stdout);                                                                                     \
            return Ret;                                                                                         \
        }                                                                                                       \
    } while (0)

#define CHECK_RET(express, name)                                                                                    \
    do {                                                                                                            \
        HI_S32 Ret;                                                                                                 \
        Ret = express;                                                                                              \
        if (HI_SUCCESS != Ret) {                                                                                    \
            printf("\033[0;31m%s failed at %s: LINE: %d with %#x!\033[0;39m\n", name, __FUNCTION__, __LINE__, Ret); \
            return Ret;                                                                                             \
        }                                                                                                           \
    } while (0)

// Function Declaration
extern aclError SdkInferDeviceContexInit(std::vector<uint32_t> &device_vec, std::vector<aclrtContext> &contex_vec);
extern aclError SdkInferDestoryRsc(std::vector<uint32_t> device_vec, std::vector<aclrtContext> contex_vec);

extern aclError SdkInferLoadModelFromMem(std::string &modelFile, uint32_t *p_modelId, aclmdlDesc **ppModelDesc,
    char **ppModelData, void **ppMem, void **ppWeight);
extern aclError SdkInferUnloadModelAndDestroyResource(uint32_t modelId, aclmdlDesc *pModelDesc, char *pModelData,
    void *pMem, void *pWeight);

extern aclError SdkInferDestroyDatasetResource(aclmdlDataset *dataset, uint32_t flag);

extern long long SdkInferElapsedus(void);
extern bool SdkInferFolderExists(std::string foldname);
extern bool SdkInferFileExists(std::string filename);
extern void SdkInferGetImgWHFromJpegBuf(unsigned char *mjpeg, uint32_t len, uint32_t &height, uint32_t &width);

extern long SdkInferGetFileSize(const char *fileName);
extern void SdkInferGetTimeStart(Time_Cost *timeCost, COST_MODULE module);
extern void SdkInferGetTimeEnd(Time_Cost *timeCost, COST_MODULE module);

#endif
