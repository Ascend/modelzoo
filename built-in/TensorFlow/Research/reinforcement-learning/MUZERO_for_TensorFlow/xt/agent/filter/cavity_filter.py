"""
cavity filter agent.
"""
import numpy as np

from xt.agent.agent import Agent
from xt.framework.register import Registers
from xt.framework.comm.message import message


@Registers.agent.register
class CavityFilter(Agent):
    """CavityFilter Agent with ddqn algorithm."""

    def infer_action(self, state, use_explore):
        """
        Infer an action with the `state`
        :param state:
        :param use_explore:
        :return: action value
        """

        state = np.reshape(state, (-1, state.shape[-1]))
        action_size = 10

        if use_explore:  # explore with remote predict
            send_data = message(state, cmd="predict")
            self.send_explorer.send(send_data)
            raw_action = self.recv_explorer.recv()
        else:  # don't explore, used in evaluate
            raw_action = self.alg.predict(state)

        a_t = np.zeros(int(action_size / 2))
        a_t[int(raw_action / 2)] = 1 if int(raw_action % 2) == 0 else -1

        # update transition data
        self.transition_data.update(
            {"cur_state": state, "raw_action": raw_action, "action": a_t,}
        )
        return a_t

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):

        action = self.transition_data["action"]
        train_action = [0] * 10
        idx = np.argmax(np.abs(action))
        val = action[idx]
        if val > 0:
            train_action[idx * 2] = 1
        else:
            train_action[idx * 2 + 1] = 1

        self.transition_data.update(
            {
                "next_state": next_raw_state,
                "train_action": train_action,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        # deliver this transition data to learner, trigger train process.
        if use_explore:
            train_data = {k: [v] for k, v in self.transition_data.items()}
            train_data = message(train_data, agent_id=self.id)
            self.send_explorer.send(train_data)

        return self.transition_data

    def sync_model(self):
        return "none"
