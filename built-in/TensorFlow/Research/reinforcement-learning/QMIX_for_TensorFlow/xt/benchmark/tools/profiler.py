# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#
# Copyright 2020 Huawei Technologies Co., Ltd
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
#!/usr/bin/env python3
"""profiler tools
    usage:

    from xt.benchmark.tools.profiler import do_profile, save_and_dump_stats
    from xt.benchmark.tools.profiler import PROFILER

    @do_profile(profiler=PROFILER)
    def to_be_profile_func():
        # do your work

    # NOTE: if here will be shutdown by os._exit()
    # we can use follows code beforce os._exit()
    # if PROFILER:
    #    save_and_dump_stats(PROFILER)
    # os._exit(0)

    default save file is 'default_stats.pkl', replace it with your likes

    we can use this script for display stats files.
    `python benchmark/tools/profiler.py -f default_stats.pkl`

"""

import argparse
import os
import pickle
import sys
from time import sleep

try:
    from line_profiler import LineProfiler, show_text
    PROFILER = LineProfiler()

    def do_profile(
            follow=[],
            profiler=None,  # pylint: disable=W0102
            stats_file="default_stats.pkl"):
        """warp the profile function into decorator"""
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler.add_function(func)
                    for sub_func in follow:
                        profiler.add_function(sub_func)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    save_and_dump_stats(profiler=profiler, stats_file=stats_file)

            return profiled_func

        return inner

except ImportError:
    PROFILER = None
    show_text = None

    def do_profile(follow=[], profiler=None):  # pylint: disable=W0102
        """dummy for import error"""
        def inner(func):
            """dummy do nothing"""
            def nothing(*args, **kwargs):
                """dummy do nothing"""
                return func(*args, **kwargs)

            return nothing

        return inner


def save_and_dump_stats(profiler, stats_file="default_stats.pkl"):
    """utils for save stats into file"""
    if not profiler:
        print("invalid profiler handler!")
        return

    if os.path.exists(stats_file):
        print("remove {}, and re-write it.".format(stats_file))
        os.remove(stats_file)
    else:
        print("write into file: {}".format(stats_file))
    try:
        # if profiler.print_stats(), will can't be dump.
        stats_info = profiler.get_stats()
        show_text(stats_info.timings, stats_info.unit)
        sys.stdout.flush()
        profiler.dump_stats(stats_file)
    # fixme: too general except
    except BaseException:
        print("profiler end without dump stats!")


def show_stats_file(stats_file):
    """utils for display stats"""
    if not show_text:
        print("Please use 'pip install line_profiler`, return with nothing do!")
        return

    def load_stats(filename):
        """ Utility function to load a pickled LineStats object from a given
        filename.
        """
        with open(filename, 'rb') as stats_handle:
            return pickle.load(stats_handle)

    print(load_stats(stats_file))
    tmp_lp = load_stats(stats_file)
    show_text(tmp_lp.timings, tmp_lp.unit)
    sleep(0.1)
    sys.stdout.flush()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="profiler tools.")

    PARSER.add_argument('-f',
                        '--stats_file',
                        nargs='+',
                        required=True,
                        help="""Read profiler stats form the (config file),
            support config file List""")

    USER_ARGS, _ = PARSER.parse_known_args()
    if _:
        print("get unkown args: {}".format(_))
    print("\nstart display with args: {} \n".format([(_arg, getattr(USER_ARGS, _arg)) for _arg in vars(USER_ARGS)]))
    print(USER_ARGS.stats_file)

    for _stats_file in USER_ARGS.stats_file:
        if not os.path.isfile(_stats_file):
            print("config file: '{}' invalid, continue!".format(_stats_file))
            continue
        show_stats_file(_stats_file)
