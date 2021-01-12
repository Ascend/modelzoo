# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from pylint import lint
import argparse

THRESHOLD = 9

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="lint for sourcecode.")

    parser.add_argument(
        "-c", "--code_path", nargs="+", default=["xt"], help="""code path""",
    )
    args, _ = parser.parse_known_args()
    print("lint with code path: ", args.code_path)

    run = lint.Run([*args.code_path,
                    "--rcfile=scripts/pylint.conf"], do_exit=False)

    score = run.linter.stats['global_note']

    if score < THRESHOLD:
        print("pylint check is failed with ", score, "which should be ", THRESHOLD)
        sys.exit(1)

    print("pylint check passed with score: {}".format(score))
