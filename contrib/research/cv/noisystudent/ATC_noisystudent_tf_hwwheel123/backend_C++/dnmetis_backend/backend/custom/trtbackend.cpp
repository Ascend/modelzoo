#include "trtbackend.h"

CREATE_BACKEND_INIT(TrtBackend,model,data)
{
  INFO_LOG("TrtBackend INIT model success");
  return 0;
}

CREATE_BACKEND_LOAD(TrtBackend,model,data)
{
  INFO_LOG("TrtBackend LOAD model success");
  return 0;
}

CREATE_BACKEND_PREDICT(TrtBackend,model,data,len,output,gpuTime)
{
  INFO_LOG("TrtBackend PREDICT model success");
  return 0;
}
CREATE_BACKEND_UNLOAD(TrtBackend,model,data)
{
  INFO_LOG("TrtBackend UNLOAD model success");
  return 0;
}


REGISTER_BASE_BACKEND_CREATOR(TRT, TrtBackend)


