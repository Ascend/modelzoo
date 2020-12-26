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

#ifndef FEATURE_ACL_H_
#define FEATURE_ACL_H_

#include "ptest.h"
#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"
#include "acl/acl_base.h"
#include "common.h"
#include "file.h"


#ifdef PLATFORM_MINI
#define L2PAGESIZE (8 * 1024 * 1024 / 64)
#define MAXSTREAMNUM 1024
#define MAXUB 256 * 1024
#elif defined(PLATFORM_CLOUD)
#define L2PAGESIZE (32 * 1024 * 1024 / 64)
#define MAXSTREAMNUM 1024
#define MAXUB 128 * 1024
#elif defined(PLATFORM_PHOENIX)
#define L2PAGESIZE (1 * 1024 * 1024 / 64)
#define MAXSTREAMNUM 64
#define MAXUB 96 * 1024
#elif defined(PLATFORM_ORLANDO)
#define L2PAGESIZE (0.5 * 1024 * 1024 / 64)
#define MAXSTREAMNUM 64
#define MAXUB 96 * 1024
#else
#define L2PAGESIZE 0
#define MAXSTREAMNUM 0
#define MAXUB 0
#endif

double getCpuTime();

class ACL : public ptest::TestFixture {
public:
    virtual void SetUp();
    virtual void TearDown();
};

class ACL_PROFILING : public ptest::TestFixture {
public:
    virtual void SetUp();
    virtual void TearDown();
};

class ACL_AI_ACL_FUNC : public ptest::TestFixture {
public:
    virtual void SetUp();
    virtual void TearDown();
};

class ACL_FUZZ : public ptest::TestFixture {
public:
    virtual void SetUp();
    virtual void TearDown();
};

class ACLConfig {
public:
    long L2OnePageSize;
    long MaxStreamNum;
    long MaxUb;
    ACLConfig()
    {
        this->L2OnePageSize = L2PAGESIZE;
        this->MaxStreamNum = MaxStreamNum;
        this->MaxUb = MaxUb;
    };
};

#endif
