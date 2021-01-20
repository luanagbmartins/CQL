# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""The entry point for running experiments with fixed replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os


import argparse

from batch_rl.fixed_replay import run_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent

from dopamine.discrete_domains import run_experiment as base_run_experiment
import tensorflow.compat.v1 as tf

from dopamine.discrete_domains import train as base_train


def create_agent(
    sess,
    environment,
    replay_data_dir,
    agent_name,
    init_checkpoint_dir,
    summary_writer=None,
):
    """Creates a DQN agent.

    Args:
      sess: A `tf.Session`object  for running associated ops.
      environment: An Atari 2600 environment.
      replay_data_dir: Directory to which log the replay buffers periodically.
      summary_writer: A Tensorflow summary writer to pass to the agent
        for in-agent training statistics in Tensorboard.

    Returns:
      A DQN agent with metrics.
    """
    if agent_name == "dqn":
        agent = dqn_agent.FixedReplayDQNAgent
    elif agent_name == "c51":
        agent = rainbow_agent.FixedReplayRainbowAgent
    elif agent_name == "quantile":
        agent = quantile_agent.FixedReplayQuantileAgent
    elif agent_name == "multi_head_dqn":
        agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
    else:
        raise ValueError("{} is not a valid agent name".format(agent_name))

    return agent(
        sess,
        num_actions=environment.action_space.n,
        replay_data_dir=replay_data_dir,
        summary_writer=summary_writer,
        init_checkpoint_dir=init_checkpoint_dir,
    )


def main(config):
    tf.logging.set_verbosity(tf.logging.INFO)
    base_run_experiment.load_gin_configs(config["gin-files"], config["gin-bindings"])
    replay_data_dir = os.path.join(config["replay-dir"], "replay_logs")
    create_agent_fn = functools.partial(
        create_agent,
        replay_data_dir=replay_data_dir,
        agent_name=config["agent-name"],
        init_checkpoint_dir=config["init-checkpoint-dir"],
    )
    runner = run_experiment.FixedReplayRunner(config["base-dir"], create_agent_fn)
    runner.run_experiment()


if __name__ == "__main__":
    config = {
        "base-dir": "/tmp/batch_rl",
        "replay-dir": "/home/luanamartins/data/RL/CQL/atari-replay-datasets/dqn/Pong/1",
        "agent-name": "quantile",
        "gin-files": [
            "/home/luanamartins/data/RL/CQL/atari/batch_rl/fixed_replay/configs/quantile.gin"
        ],
        "gin-bindings": [
            "FixedReplayRunner.num_iterations=1000",
            'atari_lib.create_atari_environment.game_name="Pong"',
            "FixedReplayQuantileAgent.minq_weight=1.0",
        ],
        "init-checkpoint-dir": "/home/luanamartins/data/RL/CQL/atari/",
    }
    main(config)
