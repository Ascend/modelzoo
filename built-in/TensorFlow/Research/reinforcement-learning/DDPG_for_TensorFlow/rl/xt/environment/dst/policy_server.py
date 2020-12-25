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
"""policy server to connect the digital sky server"""
import hashlib
import pickle
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

from xt.framework.comm.uni_comm import UniComm

IDLE_ACTION = 0


class PolicyServer(ThreadingMixIn, HTTPServer):
    """policy server work with digital sky server,
    User could send request to it"""
    def __init__(self, address, port):
        external_env = UniComm('CommByZmq', type='REQ', addr=address, port=port + 1)
        handler = _make_handler(external_env)
        HTTPServer.__init__(self, (address, port), handler)


def _make_handler(external_env):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_error(400, "Bad request.")

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length"), 0)
            raw_body = self.rfile.read(content_len)
            parsed_input = pickle.loads(raw_body)

            Key = "xthf2jn79z7y8kqe5ihlroskx5g4pnstj3ch93kvq6zplkm2vu"
            t = parsed_input["time"]
            nonce = parsed_input["nonce"]
            req_signature = parsed_input["signature"]

            signature_str = '%s|%s|%s' % (Key, nonce, t)
            m = hashlib.md5()
            m.update(bytes(signature_str, encoding='utf8'))
            signature = m.hexdigest()

            if signature == req_signature:
                try:
                    response = self.execute_command(parsed_input)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(pickle.dumps(response))
                except Exception:
                    self.send_error(500, "Sever error, please contact server provider")
                    print(traceback.format_exec())
            else:
                self.send_error(400,
                                "Specified signature is not matched with our calculation.")

        def execute_command(self, args):
            command = args["command"]
            response = {}
            if command in ["START_EPISODE", "END_EPISODE"]:
                print("start episode")
                print(args, "send data")

                external_env.send(args)
                response["episode_id"] = external_env.recv()
            elif command == "GET_ACTION":
                print("reward", args["reward"])
                external_env.send(args)
                action = external_env.recv()

                if action in args["valid_action"]:
                    response["action"] = action
                else:
                    response["action"] = IDLE_ACTION

                print("get action", action)
                print(args["episode_id"])
            elif command in ["LOG_ACTION", "LOG_RETURNS"]:
                pass
            else:
                raise Exception("Unknown command: {}".format(command))
            return response

    return Handler
