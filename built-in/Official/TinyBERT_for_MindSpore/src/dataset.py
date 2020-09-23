# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""create tinybert dataset"""

import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger

def create_tinybert_dataset(task='td', batch_size=32, device_num=1, rank=0,
                            do_shuffle="true", data_dir=None, schema_dir=None):
    """create tinybert dataset"""
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if "record" in file_name:
            data_files.append(os.path.join(data_dir, file_name))
    if task == "td":
        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    else:
        columns_list = ["input_ids", "input_mask", "segment_ids"]

    ds = de.TFRecordDataset(data_files, schema_dir, columns_list=columns_list,
                            shuffle=(do_shuffle == "true"), num_shards=device_num, shard_id=rank,
                            shard_equal_rows=True)

    ori_dataset_size = ds.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    if task == "td":
        ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    logger.info("data size: {}".format(ds.get_dataset_size()))
    logger.info("repeatcount: {}".format(ds.get_repeat_count()))

    return ds
