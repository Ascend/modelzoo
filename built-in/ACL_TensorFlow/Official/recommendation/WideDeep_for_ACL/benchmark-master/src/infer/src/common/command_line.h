/* 
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2099. All Rights Reserved.
 * Description: 命令行参数 
 * Author: Atlas
 * Create: 2020-02-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 */

#ifndef INC_COMMAND_LINE_H
#define INC_COMMAND_LINE_H
#include <gflags/gflags.h>
#include <string>

// / @brief Define flag for showing help message <br>
static const char help_message[] = "Print a usage message.";
DEFINE_bool(h, false, "Print a usage message.");
DEFINE_string(model_type, "vision", "model types only support vision/nlp/fasterrcnn/nmt/widedeep.");
DEFINE_int32(batch_size, 1, "the batch size of the model");
DEFINE_int32(device_id, 0, "the ID of the NPU device to use");
DEFINE_string(om_path, "./resnet50.om", "the path of the om model.");
DEFINE_string(input_text_path, "./tst2013.en", "the path of input text file.");
DEFINE_string(input_vocab, "./vocab.en", "the path of the vocabulary of input language.");
DEFINE_string(ref_vocab, "./vocab.vi", "the path of the vocabulary of referenced language.");
DEFINE_int32(input_width, 224, "the width of the om model input.");
DEFINE_int32(input_height, 224, "the height of the om model input.");
DEFINE_bool(output_binary, false, "Indicates whether to save the output of postprocess as binary");
/* *
 * @brief This function show a help message
 */
static void showUsage()
{
    std::cout << std::endl << "Usage: ./benchmark [Options...]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "    -h                          "
              << "show usage message." << std::endl;
    std::cout << "    -model_type                    "
              << "model types only support vision/nlp/fasterrcnn/nmt/widedeep." << std::endl;
    std::cout << "    -batch_size                    "
              << "the batch size of the model." << std::endl;
    std::cout << "    -device_id                     "
              << "the ID of the NPU device to use." << std::endl;
    std::cout << "    -om_path                       "
              << "the path of the om model." << std::endl;
    std::cout << "    -input_text_path               "
              << "the path of input text file." << std::endl;
    std::cout << "    -input_vocab                   "
              << "the path of the vocabulary of input language." << std::endl;
    std::cout << "    -ref_vocab                     "
              << "the path of the vocabulary of referenced language." << std::endl;
    std::cout << "    -input_width                   "
              << "the width of the om model." << std::endl;
    std::cout << "    -input_height                     "
              << "the height of the om model." << std::endl;
}

#endif // INC_COMMAND_LINE_H
