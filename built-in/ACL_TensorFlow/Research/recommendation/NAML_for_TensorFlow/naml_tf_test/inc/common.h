#ifndef _COMMON_H_
#define _COMMON_H_

#include <syslog.h>
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
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "ptest.h"
//#include "runtime/rt.h"
//#include "common/log_inner.h"
#define MIN(_a,_b) ((_a) > (_b) ? (_b) : (_a))
#define MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

//#include <iostream>
//using namespace std;
#define CASE_NAME ptest::TestManager::GetSingleton().GetRunningTestcaseName().c_str()
#define ESL_WAITTIME (30000)
#define FPGA_WAITTIME (30000)
#define TEST_LEVEL 6
#define MAX_UNMATCH_PRINT 10

static aclError threadError = ACL_ERROR_NONE;
static aclError error = ACL_ERROR_NONE;

#define RUN_API(expr)                                       \
    do                                                      \
    {                                                       \
        if(strstr(CASE_NAME, "FUZZ") == NULL)               \
	{ \
            printf("%s:(tid=[%lu],line=[%d])%s\n", CASE_NAME,   \
            	syscall(SYS_gettid), __LINE__, #expr);       \
	}\
        error = (expr);                                     \
    } while (0);

#define ACL_LOG(format, ...)                                             \
    do                                                                   \
    {                                                                    \
        if(strstr(CASE_NAME, "FUZZ") == NULL)\
    	 {\
            printf("%s:(tid=[%lu],line=[%d])" format "\n", CASE_NAME,        \
               syscall(SYS_gettid), __LINE__, ##__VA_ARGS__);            \
    	 }\
    } while (0);

inline void checkError(aclError error, aclError e = ACL_ERROR_NONE)
    {
        ASSERT_EQ(error, e);
    }

inline void checkAssert(aclError error, aclError e = ACL_ERROR_NONE)
    {
        printf("%s:(tid=[%lu],line=[%d]) RUN_AST failed,error[%d],expect[%d]\n", 
                CASE_NAME,syscall(SYS_gettid), __LINE__,error,e);
        ASSERT_EQ(error, e);
    }

inline void checkExpect(aclError error, aclError e = ACL_ERROR_NONE)
    {
        EXPECT_EQ(error, e);
    }

#define RUN_CHECK(expr, ...)                                    \
        RUN_API(expr);                                          \
        checkError(error, ##__VA_ARGS__);

#define RUN_ASSERT(expr, ...)                                    \
        RUN_API(expr);                                          \
        checkAssert(error, ##__VA_ARGS__);

#define RUN_EXPECT(expr, ...)                                    \
        RUN_API(expr);                                          \
        checkExpect(error, ##__VA_ARGS__);


#endif
int run_batch(char *script, char *module);
