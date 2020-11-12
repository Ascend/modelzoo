#include "aclbackend.h"


CREATE_BACKEND_INIT(AclBackend,model,data)
{
    Result ret = processSample.InitResource(model);
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }
    return SUCCESS;   
}

CREATE_BACKEND_LOAD(AclBackend,model,data)
{
    INFO_LOG("AclBackend LOAD model success");
    return SUCCESS;
}

CREATE_BACKEND_PREDICT(AclBackend,model,data,len,output,npuTime)
{
    Result ret = processSample.Process(data,len,output,npuTime);
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }
    INFO_LOG("execute sample success");
    return SUCCESS;
}

CREATE_BACKEND_UNLOAD(AclBackend,model,data)
{
     Result ret = processSample.Unload();
    if (ret != SUCCESS) {
        ERROR_LOG("sample unload failed");
        return FAILED;
    }
    INFO_LOG("AclBackend UNLOAD model success");
    return SUCCESS;
}



REGISTER_BASE_BACKEND_CREATOR(ACL, AclBackend)


