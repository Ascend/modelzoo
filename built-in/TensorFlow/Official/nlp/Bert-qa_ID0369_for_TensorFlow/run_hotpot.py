# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from npu_bridge.npu_init import *
import tqdm


def create_example_dict(context, answer_start, answer, id, is_impossible, question):
    return {
        "context": context,
        "qas": [
            {
                "answers": [{"answer_start": answer_start, "text": answer}],
                "id": id,
                "is_impossible": is_impossible,
                "question": question,
            }
        ],
    }


def create_para_dict(example_dicts):
    if type(example_dicts) == dict:
        example_dicts = [example_dicts]
    return {"paragraphs": example_dicts}


def add_yes_no(string):
    # Allow model to explicitly select yes/no from text (location front, avoid truncation)
    return " ".join(["yes", "no", string])


def convert_hotpot_to_squad_format(
    json_dict, gold_paras_only=True, combine_context=True
):
    new_dict = {"data": []}
    count = 0
    for example in json_dict:
        raw_contexts = example["context"]

        if gold_paras_only:
            support = {
                para_title: line_num
                for para_title, line_num in example["supporting_facts"]
            }
            raw_contexts = [lst for lst in raw_contexts if lst[0] in support]

        contexts = ["".join(lst[1]) for lst in raw_contexts]
        if combine_context:
            contexts = [" ".join(contexts)]

        answer = example["answer"]
        for context in contexts:
            context = add_yes_no(context)
            answer_start = context.index(answer) if answer in context else -1

            new_dict["data"].append(
                create_para_dict(
                    create_example_dict(
                        context=context,
                        answer_start=answer_start,
                        answer=answer,
                        id=str(count),  # SquadExample.__repr__ only accepts type==str
                        is_impossible=(answer_start == -1),
                        question=example["question"],
                    )
                )
            )
            count += 1
    return new_dict


def convert_hotpot_to_squad_format_classify_support(json_dict):
    # Not training model to find answer, but just to "classify" support-ness
    new_dict = {"data": []}
    count = 0
    for example in json_dict:
        support = {
            para_title: line_num for para_title, line_num in example["supporting_facts"]
        }

        for para_title, para_lines in example["context"]:
            answer = {True: "yes", False: "no"}[para_title in support]
            context = add_yes_no(" ".join(para_lines))
            answer_start = context.index(answer) if answer in context else -1

            new_dict["data"].append(
                create_para_dict(
                    create_example_dict(
                        context=context,
                        answer_start=answer_start,
                        answer=answer,
                        id=str(count),  # SquadExample.__repr__ only accepts type==str
                        is_impossible=(answer_start == -1),
                        question=example["question"],
                    )
                )
            )
            count += 1
    return new_dict


def convert_hotpot_to_squad_format_classify_support_double(json_dict):
    # Not training model to find answer, but just to "classify" support-ness
    new_dict = {"data": []}
    count = 0
    for example in tqdm.tqdm(json_dict):
        support = {
            para_title: line_num for para_title, line_num in example["supporting_facts"]
        }

        n_context_paras = len(example["context"])
        for i in range(n_context_paras):
            para_title_i, para_lines_i = example["context"][i]
            if para_title_i in support:
                for j in range(n_context_paras):
                    if i != j:
                        para_title_j, para_lines_j = example["context"][j]
                        answer = {True: "yes", False: "no"}[para_title_j in support]
                        combined = " ".join(para_lines_i) + " " + " ".join(para_lines_j)
                        context = add_yes_no(combined)
                        answer_start = context.index(answer)

                        new_dict["data"].append(
                            create_para_dict(
                                create_example_dict(
                                    context=context,
                                    answer_start=answer_start,
                                    answer=answer,
                                    id=str(
                                        count
                                    ),  # SquadExample.__repr__ only accepts type==str
                                    is_impossible=(answer_start == -1),
                                    question=example["question"],
                                )
                            )
                        )
                        count += 1
    return new_dict
