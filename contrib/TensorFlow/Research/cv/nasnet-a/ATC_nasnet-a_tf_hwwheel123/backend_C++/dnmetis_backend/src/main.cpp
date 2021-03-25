#include <pybind11/pybind11.h>
#include "BackendFactory.h"
#include "BaseBackend.h"
//#include "common.h"
#include "utils.h"
#include <string>

using namespace std;

bool warmup=true;

long deviceTime=0;

std::shared_ptr<BaseBackend> backend;
Config * Config::instance = nullptr;

int backend_setconfig(char* cfg)
{
   Config::setInstance(cfg);
}

int backend_load(int backend_type,char* omModelPath,char* binfile)
{
   backend = BackendFactory::Instance()->CreateBaseBackend(FrameworkType(backend_type));
   if(backend == nullptr)
   {
      ERROR_LOG("FAILED, Not found the test backend, type:%d.", FrameworkType(backend_type));
      return FAILED;
   }
   backend->init(omModelPath,binfile);
   printf("[INFO] AclBackend init OK\n");
   backend->load(omModelPath,binfile);
   printf("[INFO] AclBackend load OK\n");
}

//py::array backend_predict(int type, char* omModelPath, py::array binfile)
vector<py::array> backend_predict(int type, char* omModelPath, py::array binfile)
{
   if(warmup)
   {
      printf("[INFO] start warmup AclBackend predict\n");
      //warmup=false;
   }
   INFO_LOG("start backend_predict is %d", Utils::getCurrentTime());
   std::vector<Output_buf> result_buf;
   //INFO_LOG("binfile.nbytes is %d", binfile.nbytes());
   deviceTime = 0;
   backend->predict(omModelPath, binfile.mutable_data(), binfile.nbytes(),result_buf, deviceTime);
   INFO_LOG("Pure device execute time is %f ms", deviceTime);
   if(warmup)
   {
      printf("[INFO] end warmup AclBackend predict\n");
      warmup=false;
   }

   INFO_LOG("end backend_predict is %d", Utils::getCurrentTime());

   vector<py::array> vec_result;
   for(int i =0 ; i<result_buf.size();i++)
   {
       std::string str;
       if(!result_buf[i].format.compare("uint8"))
           str=py::format_descriptor<uint8_t>::format();
       if(!result_buf[i].format.compare("int8"))
           str=py::format_descriptor<int8_t>::format();
       if(!result_buf[i].format.compare("float"))
           str=py::format_descriptor<float>::format();
       if(!result_buf[i].format.compare("float16"))
           str=py::format_descriptor<float16>::format();
       if(!result_buf[i].format.compare("int64"))
           str=py::format_descriptor<int64_t>::format();
       if(!result_buf[i].format.compare("uint64"))
           str=py::format_descriptor<uint64_t>::format();
       py::buffer_info tmp=py::buffer_info(
	           result_buf[i].ptr,
		       (ssize_t)result_buf[i].itemsize, //itemsize
		       str,
		       (ssize_t)result_buf[i].ndim,// ndim
		       result_buf[i].shape, // shape
		       result_buf[i].strides  //strides
           );
       py::dtype dt = py::dtype(str);
       py::array result = py::array(dt,tmp.shape, tmp.strides, tmp.ptr);
       vec_result.push_back(result);
   }
   return vec_result;
   //return result;
}

long backend_get_device_time()
{
   return deviceTime;
}

int backend_unload(int type,char* omModelPath,char* binfile)
{
   backend->unload(omModelPath,binfile);
   printf("[INFO] AclBackend unload OK\n");
}

namespace py = pybind11;

PYBIND11_MODULE(dnmetis_backend, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: dnmetis_backend

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    /*m.def("backend_main", &backend_main, R"pbdoc(
        backend
    )pbdoc");*/
    m.def("backend_setconfig", &backend_setconfig, R"pbdoc(
        backend
    )pbdoc");

    m.def("backend_load", &backend_load, R"pbdoc(
        backend
    )pbdoc");

    m.def("backend_predict", &backend_predict, R"pbdoc(
        backend
    )pbdoc");

    m.def("backend_get_device_time", &backend_get_device_time, R"pbdoc(
        backend
    )pbdoc");

    m.def("backend_unload", &backend_unload, R"pbdoc(
        backend
    )pbdoc");
    m.def("add", [](int i, int j) { return i + j; }, R"pbdoc(
        add
    )pbdoc");
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        subtract
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
