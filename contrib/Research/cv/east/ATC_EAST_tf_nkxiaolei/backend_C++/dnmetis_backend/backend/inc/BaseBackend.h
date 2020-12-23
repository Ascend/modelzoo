
/**
 * Copyright 2020 Huawei Technologies Co., Ltd

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

#ifndef BASE_BACKEND_H_
#define BASE_BACKEND_H_

//#include "utils.h"
#include "common.h"
#include <vector>
using namespace std;

class BaseBackend {
 public:
  BaseBackend() {};
  /**
   * @ingroup domi_omg
   * @brief Deconstructor
   */
  ~BaseBackend() {};

  virtual int init(char* model,char* data){};

  virtual int load(char* model,char* data){};

  virtual int predict(char* model,void* data,int len, std::vector<Output_buf> &output, long &deviceTime){};
 
  virtual int unload(char* model,char* data){};

  int param();

  int runner();

  int runnerthread();

  int statistic();

};



#define CREATE_BACKEND(x)  \
class x : public BaseBackend{ \
public:                       \
        x(){};                  \
        ~x(){};       \
       int init(char* model,char* data);   \
       int load(char* model,char* data);      \
       int predict(char* model,void* data,int len, vector<Output_buf> &output, long &deviceTime);  \
       int unload(char* model,char* data);   \
       char* model;   \
       char* data;     \
 }; \

#define CREATE_BACKEND_INIT(x,model,data)  \
   int x::init(char* model,char* data)
#define CREATE_BACKEND_LOAD(x,model,data)   \
   int x::load(char* model,char* data)
#define CREATE_BACKEND_PREDICT(x,model,data,len,output,deviceTime)   \
   int x::predict(char* model,void* data,int len,vector<Output_buf> &output, long &deviceTime)
#define CREATE_BACKEND_UNLOAD(x,model,data) \
   int x::unload(char* model,char* data)

#endif 

