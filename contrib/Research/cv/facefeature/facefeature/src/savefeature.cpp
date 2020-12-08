#include "savefeature.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <fstream>

Result save(FaceInfo &faceInfo) {
    fstream outputFile("../faceLib/people.bin", ios::out | ios::binary| ios::app);
    if (!outputFile)
    {
        INFO_LOG("Error writeFaceLib opening file. Program aborting.\n");
        return FAILED;
    }
    outputFile.write(reinterpret_cast<char *>(&faceInfo), sizeof(FaceInfo));
    outputFile.close();
    return SUCCESS;
}

Result readFaceLib(std::vector<FaceInfo> &faceInfos) {
    fstream inputFile("../faceLib/people.bin", ios::in | ios::binary);
    if (!inputFile)
    {
        INFO_LOG("Error readFaceLib opening file. Program aborting.\n");
        return FAILED;
    }
    FaceInfo faceInfo;
    inputFile.read(reinterpret_cast<char *>(&faceInfo), sizeof (FaceInfo));
    while (!inputFile.eof())
    {
        faceInfos.emplace_back(faceInfo);
        inputFile.read(reinterpret_cast<char *>(&faceInfo), sizeof(FaceInfo));
    }
    inputFile.close();
    return SUCCESS;
}