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
"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
import datetime
import logging
import time
import os
import numpy as np


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict < str, Any >
            contains returns, average speed per step
        """
        num_steps = self.env.env_params.horizon

        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        # used to store
        info_dict = {
            "returns": [],
            "velocities": [],
            "outflows": [],
        }
        info_dict.update({
            key: [] for key in self.custom_callables.keys()
        })

        if rl_actions is None:
            def rl_actions(*_):
                return None

        # time profiling information
        t = time.time()
        times = []

        for i in range(num_runs):
            ret = 0
            vel = []
            custom_vals = {key: [] for key in self.custom_callables.keys()}
            state = self.env.reset()
            for j in range(num_steps):
                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))

                # Compute the velocity speeds and cumulative returns.
                veh_ids = self.env.k.vehicle.get_ids()
                vel.append(np.mean(self.env.k.vehicle.get_speed(veh_ids)))
                ret += reward

                # Compute the results for the custom callables.
                for (key, lambda_func) in self.custom_callables.items():
                    custom_vals[key].append(lambda_func(self.env))

                if done:
                    break

            # Store the information from the run in info_dict.
            outflow = self.env.k.vehicle.get_outflow_rate(int(500))
            info_dict["returns"].append(ret)
            info_dict["velocities"].append(np.mean(vel))
            info_dict["outflows"].append(outflow)
            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))

            print("Round {0}, return: {1}".format(i, ret))

        # Print the averages/std for all variables in the info_dict.
        for key in info_dict.keys():
            print("Average, std {}: {}, {}".format(
                key, np.mean(info_dict[key]), np.std(info_dict[key])))

        print("Total time:", time.time() - t)
        print("steps/second:", np.mean(times))
        self.env.terminate()

        if convert_to_csv and self.env.simulator == "traci":
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = \
                "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict
