/**
* @file common.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>
#include<vector>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "half.hpp"
#include "Config.h"

#define LOG 2

#define INFO_LOG(fmt, args...)   if(Config::getInstance()->Read("backend_loglevel", 0)>=3) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...)   if(Config::getInstance()->Read("backend_loglevel", 0)>=2) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...)  if(Config::getInstance()->Read("backend_loglevel", 0)>=1) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

namespace py = pybind11;
// half_float::half behaviors like float, but with different precision

using float16 = half_float::half;

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <>
struct type_caster<float16> {
public:
    PYBIND11_TYPE_CASTER(float16, _("float16"));
    using float_caster = type_caster<float>;

    bool load(handle src, bool convert) {
        float_caster caster;
        if (caster.load(src, convert)) {
            this->value = float16(float(caster));  // Implicit cast defined by `type_caster`.
            return true;

        }
        return false;
    }
    static handle cast(float16 src, return_value_policy policy, handle parent) {
        return float_caster::cast(float(src), policy, parent);
    }
};

constexpr int NPY_FLOAT16 = 23;

template <>
struct npy_format_descriptor<float16> {
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
  static constexpr auto name() {
    return _("float16");
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)



typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

enum FrameworkType {
  CAFFE = 0,
  MINDSPORE = 1,
  TENSORFLOW = 3,
  ANDROID_NN,
  ACL,
  TRT,
  HIAI_ENGINE,
  FRAMEWORK_RESERVED,
};

struct Output_buf {
    void *ptr = nullptr;          // Pointer to the underlying storage
    int64_t itemsize = 0;         // Size of individual items in bytes
    int64_t size = 0;             // Total number of entries
    std::string format;           // For homogeneous buffers, this should be set to format_descriptor<T>::format()
    int64_t ndim = 0;             // Number of dimensions
    std::vector<int64_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<int64_t> strides; // Number of bytes between adjacent entries (for each per dimension)
    bool readonly = false;        // flag to indicate if the underlying storage may be written to
};


