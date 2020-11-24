# 1. remove not released code in agent module
rm  -r ../xt/agent/cartpole
rm  -r ../xt/agent/controller
rm  -r ../xt/agent/filter
rm  -r ../xt/agent/muzero
rm  -r ../xt/agent/overtake
rm  -r ../xt/agent/qmix
rm ../xt/agent/ppo/figure8_ppo.py
rm ../xt/agent/ppo/catchpigs_ppo.py
rm ../xt/agent/impala/vector_atari_impala_tf.py
rm ../xt/agent/impala/cartpole_impala_tf.py
rm ../xt/agent/impala/atari_impala_tf.py
rm ../xt/agent/dqn/sumo_dqn.py
rm ../xt/agent/dqn/server_agent.py

# 2. remove not released code in algorithm module
rm  -r ../xt/algorithm/ddpg
rm  -r ../xt/algorithm/gail
rm  -r ../xt/algorithm/muzero
rm  -r ../xt/algorithm/qmix
rm  -r ../xt/algorithm/reinforce
rm ../xt/algorithm/ppo/ppo_share_weights.py
rm ../xt/algorithm/impala/impala_tf.py
rm ../xt/algorithm/dqn/rainbow.py
rm ../xt/algorithm/dqn/dqn_sumo.py
rm ../xt/algorithm/dqn/dqn_pri.py
rm ../xt/algorithm/dqn/ddqn.py
rm ../xt/algorithm/segment_tree.py
rm ../xt/algorithm/prioritized_replay_buffer.py

# 3. remove not released code in environment module
rm  -r ../xt/environment/dst
rm  -r ../xt/environment/flow
rm  -r ../xt/environment/ma
rm  -r ../xt/environment/rl_simu
rm  -r ../xt/environment/sumo
# 4. remove not released code in model module
rm  -r ../xt/model/ddpg
rm  -r ../xt/model/gail
rm  -r ../xt/model/muzero
rm  -r ../xt/model/qmix
rm  -r ../xt/model/reinforce
rm ../xt/model/model_npu.py

rm ../xt/model/dqn/rainbow_network_cnn.py
rm ../xt/model/dqn/rainbow_network_mlp.py
rm ../xt/model/dqn/ddq_network.py
rm ../xt/model/dqn/ddq_network_sumo.py

rm ../xt/model/impala/vtrace_tf.py
rm ../xt/model/impala/impala_network_mlp.py
rm ../xt/model/impala/impala_network_cnn_small.py
rm ../xt/model/impala/impala_network_cnn.py
rm ../xt/model/impala/impala_fcnet_tf.py
rm ../xt/model/impala/impala_cnn_with_logits.py
rm ../xt/model/impala/impala_cnn_small_tf.py

rm ../xt/model/ppo/ppo_cnn_tf_small.py
rm ../xt/model/ppo/ppo_cnn_tf.py
rm ../xt/model/ppo/ppo_cnn_small.py
rm ../xt/model/ppo/ppo_cnn_pigs.py
rm ../xt/model/ppo/actor_critic_ppo_cnn.py
rm ../xt/model/ppo/actor_critic_ppo.py
#5. remove  ci and cython file
rm -f ../.gitlab-ci.yml
rm -f ../setup_cython.py
rm -f ../build.py
#6 modify config file
rm -r ../examples/benchmark_cases
rm -r ../examples/default_cases
rm -r ../examples/ma_cases
#7 remove test file
rm -r ../tests/qmix
rm -r ../tests/test_CatchPigs.py
#8 remove git git file
rm -rf ../.git

# 9 rm un-used docs
rm -rf ../docs/third
rm -rf ../docs/.images/overtake.gif
rm -rf ../docs/.images/speed.gif
rm -rf ../docs/.images/speer.gif

# remove self
rm ../scripts/opensource_0.1.0.sh
