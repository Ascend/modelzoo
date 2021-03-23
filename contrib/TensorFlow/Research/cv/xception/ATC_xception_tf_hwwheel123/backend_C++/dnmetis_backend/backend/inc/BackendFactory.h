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

#ifndef BASE_BACKEND_FACTORY_H_
#define BASE_BACKEND_FACTORY_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "BaseBackend.h"



typedef std::shared_ptr<BaseBackend> (*BASE_BACKEND_CREATOR_FUN)(void);

class BackendFactory {
 public:
  static BackendFactory *Instance();
  int Init();
  /**
  * @ingroup domi_omg
  * @brief Create a backend based on the type entered
  * @param [in] type Framework type
  * @return Created backend
  */
  std::shared_ptr<BaseBackend> CreateBaseBackend(const FrameworkType type);

  /**
  * @ingroup domi_omg
  * @brief Register create function
  * @param [in] type Framework type
  * @param [in] fun ModelParser's create function
  */
  void RegisterCreator(const FrameworkType type, BASE_BACKEND_CREATOR_FUN fun);

 protected:
  BackendFactory() {}
  ~BackendFactory();

 private:
  std::map<FrameworkType, BASE_BACKEND_CREATOR_FUN> creator_map_;
  bool is_init_ = false;
};  // end class BackendFactory

class BaseBackendRegisterar {
 public:
  BaseBackendRegisterar(const FrameworkType type, BASE_BACKEND_CREATOR_FUN fun) {
    BackendFactory::Instance()->RegisterCreator(type, fun);
  }
  ~BaseBackendRegisterar() {}
};

// Registration macros for backend
typedef std::shared_ptr<BaseBackend> (*BASE_BACKEND_CREATOR_FUN)(void);

// Registration macros for BaseBackend
#define REGISTER_BASE_BACKEND_CREATOR(type, clazz)               \
  std::shared_ptr<BaseBackend> Creator_##type##_Base_Backend() { \
    std::shared_ptr<clazz> ptr = nullptr;                             \
    try {                                                        \
      ptr = std::make_shared<clazz>();                                \
    } catch (...) {                                              \
      ptr = nullptr;                                             \
    }                                                            \
    return std::shared_ptr<BaseBackend>(ptr);                    \
  }                                                              \
  BaseBackendRegisterar g_##type##_Base_Backend_Creator(type, Creator_##type##_Base_Backend);


#endif 







