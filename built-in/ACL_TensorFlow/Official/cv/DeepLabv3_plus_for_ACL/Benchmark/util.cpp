/* *
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* */

#include "util.h"
#include <unordered_map>
#include <cstring>
#include <algorithm>
#if 0
static std::unordered_map<aclError, std::string> errorMap = {
	{ACL_ERROR_NONE, "success"},
	{ACL_ERROR_INVALID_PARAM, "params may not valid"},
	{ACL_ERROR_BAD_ALLOC, "alloc memory failed"},
	{ACL_ERROR_RT_FAILURE, "runtime failure"},
	{ACL_ERROR_GE_FAILURE, "GE failure"},
	{ACL_ERROR_OP_NOT_FOUND, "OP not find"},
	{ACL_ERROR_OP_LOAD_FAILED, "OP loads failed"},
	{ACL_ERROR_READ_MODEL_FAILURE, "load model failed"},
	{ACL_ERROR_PARSE_MODEL, "parse model failed"},
	{ACL_ERROR_MODEL_MISSING_ATTR, "model misssing attr"},
	{ACL_ERROR_DESERIALIZE_MODEL, "deserilize model failed"},
	//	{ACL_ERROR_MULTIPLE_MODEL_MATCHED, "multiple model matched"},
	//{ACL_ERROR_EVENT_NOT_READY, "event not ready"},
	//{ACL_ERROR_EVENT_COMPLETE, "event not complete"},
	{ACL_ERROR_UNSUPPORTED_DATA_TYPE, "unsupport datatype"},
	{ACL_ERROR_REPEAT_INITIALIZE, "initial repeated"},
	//{ACL_ERROR_COMPILER_NOT_REGISTERED, "compilter not registered"},
	{ACL_ERROR_PATH_INVALID, "path invalid"},
	{ACL_ERROR_PARSE_PARAM_FAILED, "parse params failed"},
	{ACL_ERROR_DVPP_ERROR, "dvpp errors"}
};


std::string CausedBy(aclError error)
{
	return errorMap[error];
}
#endif

bool FolderExists(std::string foldname)
{
	DIR* dir;
	if ((dir = opendir(foldname.c_str())) == NULL) {
		return false;
	}
	closedir(dir);
	return true;
}

void* ReadFile(std::string fileLocation, uint64_t &fileSize)
{
    aclError ret;
    FILE *pFile = fopen(fileLocation.c_str(), "r");
    if (pFile == nullptr) {
        LOG("open file %s failed\n", fileLocation.c_str());
        return nullptr;
    }

    fseek(pFile, 0, SEEK_END);
    fileSize = ftell(pFile);

    void *buff = nullptr;
    ret = aclrtMallocHost(&buff, fileSize);
    if (ret != ACL_ERROR_NONE) {
        LOG("Malloc host buff failed[%d]\n", ret);
        return nullptr;
    }

    rewind(pFile);
    fread(buff, sizeof(char), fileSize, pFile);
    fclose(pFile);
	return buff;
}

bool FileExists(std::string filename)
{
	std::fstream file;
	file.open(filename, std::ios::in);
	if (!file) {
		return false;
	}
	
	file.close();
	return true;
}

char* ReadBinFile(std::string fileName, uint32_t& fileSize)
{
	std::ifstream binFile(fileName, std::ifstream::binary);

	if (binFile.is_open() == false) {
		LOG("open file[%s] failed\n", fileName.c_str());
		return nullptr;
	}

	binFile.seekg(0, binFile.end);
	uint32_t binFileBufferLen = binFile.tellg();

	if (binFileBufferLen == 0) {
		LOG("binfile is empty, filename: %s", fileName.c_str());
		binFile.close();
		return nullptr;
	}

	binFile.seekg(0, binFile.beg);
	char* binFileBufferData = new(std::nothrow) char[binFileBufferLen];
	LOG("binFileBufferData:%p\n", binFileBufferData);

	if (binFileBufferData == nullptr) {
		LOG("malloc binFileBufferData failed\n");
		binFile.close();
		return nullptr;
	}

	binFile.read(binFileBufferData, binFileBufferLen);
	binFile.close();
	fileSize = binFileBufferLen;
	return binFileBufferData;
}

aclError GetFiles(std::string path, std::vector<std::string>& files)
{
	DIR* dir;
	struct dirent* ptr;
	char base[1000];

	if ((dir = opendir(path.c_str())) == NULL) {
		LOG("Open dir %s error.\n", path.c_str());
		return ACL_ERROR_PATH_INVALID;
	}

	while ((ptr = readdir(dir)) != NULL) {
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
			//current dir OR parrent dir
			continue;
		} else if (ptr->d_type == 8) {
			//file
			files.push_back(ptr->d_name);
		} else if (ptr->d_type == 10) {
			//link file
			continue;
		} else if (ptr->d_type == 4) {
			//dir
			continue;
		}
	}

	closedir(dir);
	std::sort(files.begin(), files.end());
	return ACL_ERROR_NONE;
}

aclError FreeDevMemory(aclmdlDataset* dataset)
{
    aclError ret;
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        aclrtFree(data);
        aclDestroyDataBuffer(dataBuffer);
    }

	return ACL_ERROR_NONE;
}

aclError DestroyDatasetResurce(aclmdlDataset* dataset, uint32_t flag)
{
    aclError ret = ACL_ERROR_NONE;

    if (nullptr == dataset) {
        LOG("dataset == null\n");
        return 1;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        if (nullptr == dataBuffer) {
            LOG("dataBuffer == null\n");
            continue;
        }

        void* data = aclGetDataBufferAddr(dataBuffer);
        if (nullptr != data) {
            if (1 == flag) {
                if (i > 0) {
                    ret = aclrtFree(data);
                    if (ret != ACL_ERROR_NONE) {
                        LOG("aclrtFree data failed, ret %d\n", ret);
                    }
                } else {
                    ret = acldvppFree(data);
                    if (ret != ACL_ERROR_NONE) {
                        LOG("acldvppFree data failed, ret %d\n", ret);
                    }
                }
            } else {
                ret = aclrtFree(data);
                if (ret != ACL_ERROR_NONE) {
                    LOG("aclrtFree data failed, ret %d\n", ret);
                }
            }
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            LOG("Destroy dataBuffer failed, ret %d\n", ret);
        }
    }

    ret = aclmdlDestroyDataset(dataset);
    if (ret != ACL_ERROR_NONE) {
        LOG("aclrtFree dataset failed, ret %d\n", ret);
    }

    return ret;
}


