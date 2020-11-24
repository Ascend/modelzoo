# -*- coding:utf-8 -*-
"""Test function for `vega.core.common.file_ops.copy`"""
import unittest
import os
from vega.core.common import FileOps


class TestDataset(unittest.TestCase):

    def test_copy(self):
        file_dir = os.path.abspath(os.path.dirname(__file__))
        new_dir = file_dir + '/automl'
        os.mkdir(new_dir)
        open(os.path.join(file_dir, 'test1.txt'), 'a').close()
        src_file = os.path.join(file_dir, 'test1.txt')
        dst_file = os.path.join(new_dir, "test1.txt")
        FileOps.copy_file(src_file, dst_file)
        self.assertEqual(os.path.isfile(dst_file), True)
        new_folder = file_dir + "/new"
        FileOps.copy_folder(new_dir, new_folder)
        file_num = len([x for x in os.listdir(new_folder)])
        self.assertEqual(file_num, 1)


if __name__ == "__main__":
    unittest.main()
