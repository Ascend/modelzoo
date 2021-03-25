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

#include "BackendFactory.h"

BackendFactory *BackendFactory::Instance() {
  static BackendFactory instance;
  return &instance;
}

int BackendFactory::Init() {
  if (!is_init_) {
    std::string skt_bin = "libaclbackend.so";
    //handle_ = dlopen(skt_bin.c_str(), RTLD_NOW | RTLD_GLOBAL);
    //if (handle_ == nullptr) {
    //  GELOGE(FAILED, "SKT: open skt lib failed, please check LD_LIBRARY_PATH.");
    //}
  }
  is_init_ = true;

  return 0;
}


std::shared_ptr<BaseBackend> BackendFactory::CreateBaseBackend(const FrameworkType type) {
  std::map<FrameworkType, BASE_BACKEND_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }

  return nullptr;
}

void BackendFactory::RegisterCreator(const FrameworkType type, BASE_BACKEND_CREATOR_FUN fun) {
  std::map<FrameworkType, BASE_BACKEND_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return;
  }

  creator_map_[type] = fun;
}

BackendFactory::~BackendFactory() {
  creator_map_.clear();
}



