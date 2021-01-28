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

import ast
import argparse
import numpy as np

from batch_rl.fixed_replay.environments import ACPulse
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
        observation_shape=environment.observation_shape,
        observation_dtype=environment.observation_dtype,
        replay_data_dir=replay_data_dir,
        summary_writer=summary_writer,
        init_checkpoint_dir=init_checkpoint_dir,
    )


def create_environment(
    observation_shape=(80,),
    observation_dtype=np.float32,
):
    env = ACPulse(observation_shape, observation_dtype)
    return env


def main(configs):
    tf.logging.set_verbosity(tf.logging.INFO)
    base_run_experiment.load_gin_configs(configs.gin_files, configs.gin_bindings)
    replay_data_dir = os.path.join(configs.replay_dir, "replay_logs")
    create_agent_fn = functools.partial(
        create_agent,
        replay_data_dir=replay_data_dir,
        agent_name=configs.agent_name,
        init_checkpoint_dir=configs.init_checkpoint_dir,
    )
    create_environment_fn = functools.partial(create_environment)
    runner = run_experiment.FixedReplayRunner(
        configs.base_dir, create_agent_fn, create_environment_fn=create_environment_fn
    )

    dataset_path = os.path.join(
        os.path.realpath("."), "data/processed/v5_dataset/test_dataset_users/"
    )
    chkpt_path = os.path.join(
        os.path.realpath("."), "models/reward_pred_v0_model/release/80_input"
    )
    runner.set_offline_evaluation(dataset_path, chkpt_path)

    runner.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CQL")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=os.path.join(
            os.path.realpath("."), "cql-dataset/v5_scoremax_dataset_cql"
        ),
    )
    parser.add_argument(
        "--base-dir", type=str, default=os.path.join(os.path.realpath("."), "runs")
    )
    parser.add_argument("--agent-name", type=str, default="quantile")
    parser.add_argument(
        "--gin-files",
        type=str,
        default=str(
            [
                os.path.join(
                    os.path.realpath("."), "batch_rl/fixed_replay/configs/quantile.gin"
                )
            ]
        ),
    )
    parser.add_argument("--gin-bindings", type=str, default="[]")
    parser.add_argument(
        "--init-checkpoint-dir",
        type=str,
        default=os.path.join(os.path.realpath("."), "runs"),
    )

    args = parser.parse_args()
    args.gin_files = ast.literal_eval(args.gin_files)
    args.gin_bindings = ast.literal_eval(args.gin_bindings)

    main(args)
