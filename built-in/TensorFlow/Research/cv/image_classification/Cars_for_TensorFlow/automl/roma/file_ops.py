# -*- coding:utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

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

"""RomaFileOps class."""
import os
import fcntl
import logging
import moxing
from vega.core import FileOps


class RomaFileOps(object):
    """This is the class which use the function of roma."""

    @classmethod
    def copy_folder(cls, src, dst):
        """Copy a local dir to s3 bucket.

        :param str src: path to local directory.
        :param str dst: path to s3 directory. eg. `s3://bucket/upload/`

        """
        if dst is None or dst == "":
            return
        if moxing.file.exists(src):
            moxing.file.copy_parallel(src, dst)
        else:
            logging.warn("remote path {} doesn't exist".format(src))

    @classmethod
    def copy_file(cls, src, dst):
        """Copy a local file to s3 bucket.

        :param str src: path to local file.
        :param str dst: path to s3 file. eg. `s3://bucket/upload/a.caffemodel`

        """
        if dst is None or dst == "":
            return
        if moxing.file.exists(src):
            moxing.file.copy(src, dst)
        else:
            logging.warn("remote file {} doesn't exist".format(src))

    @classmethod
    def download_dataset(cls, src_path):
        """If the data path is s3 path, then it will copy the data to /cache.

        and the path will be changed to cache path.
        :param src_path: the data path
        :type src_path: str
        :raises FileNotFoundError: if the file path is not exist, an error will raise
        :return: the final data path
        :rtype: str
        """
        if src_path.split(':')[0] == 's3':
            if src_path[-1] == '/':
                src_path = src_path[0:-1]
            cache_path = '/cache/' + src_path[5:]
            # set a signal file
            src_name = src_path[5:].replace('/', '_')
            signal_file = os.path.join('/cache', 'DATASET_COPY_OK_{}'.format(src_name))
            if not os.path.isfile(signal_file):
                with open(signal_file, 'w') as fp:
                    fp.write('{}'.format(0))
            with open(signal_file, 'r+') as fp:
                # use file lock to make sure only download dataset once on same node
                fcntl.flock(fp, fcntl.LOCK_EX)
                signal = int(fp.readline().strip())
                if signal == 0:
                    if moxing.file.exists(src_path):
                        logging.info("Copy the data to cache.")
                        moxing.file.copy_parallel(src_path, cache_path)
                    else:
                        raise FileNotFoundError(src_path + 'is not existed')
                    with open(signal_file, 'w') as fn:
                        fn.write('{}'.format(1))
                else:
                    logging.info("The data has already existed in the cache.")
                fcntl.flock(fp, fcntl.LOCK_UN)
            final_path = cache_path
        else:
            if os.path.exists(src_path):
                final_path = src_path
            else:
                raise FileNotFoundError(src_path + 'is not existed')
        return final_path

    @classmethod
    def exists(cls, path):
        """Is folder existed or not.

        :param folder: folder
        :type folder: str
        :return: folder existed or not.
        :rtype: bool
        """
        return moxing.file.exists(path)


def replace_file_ops():
    """Replace method of Fileops with RomaFileOps."""
    FileOps.copy_folder = RomaFileOps.copy_folder
    FileOps.copy_file = RomaFileOps.copy_file
    FileOps.download_dataset = RomaFileOps.download_dataset
    FileOps.exists = RomaFileOps.exists
