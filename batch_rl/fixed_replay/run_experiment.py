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

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import torch
import pandas as pd
import numpy as np
from statistics import mean
from ray.rllib.offline.json_reader import JsonReader
from batch_rl.fixed_replay.reward_predictor import RewardPredictor

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
            checkpoint_file_prefix
        )

        # Code for the loading a checkpoint at initialization
        init_checkpoint_dir = (
            self._agent._init_checkpoint_dir
        )  # pylint: disable=protected-access
        if (self._start_iteration == 0) and (init_checkpoint_dir is not None):
            if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir, checkpoint_file_prefix
                )
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir
                )
                if latest_init_checkpoint >= 0:
                    experiment_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint
                    )
                    if self._agent.unbundle(
                        init_checkpoint_dir, latest_init_checkpoint, experiment_data
                    ):
                        if experiment_data is not None:
                            assert "logs" in experiment_data
                            assert "current_iteration" in experiment_data
                            self._logger.data = experiment_data["logs"]
                            self._start_iteration = (
                                experiment_data["current_iteration"] + 1
                            )
                        tf.logging.info(
                            "Reloaded checkpoint from %s and will start from iteration %d",
                            init_checkpoint_dir,
                            self._start_iteration,
                        )

    def _run_train_phase(self):
        """Run training phase."""
        self._agent.eval_mode = False
        start_time = time.time()
        for _ in range(self._training_steps):
            self._agent._train_step()  # pylint: disable=protected-access
        time_delta = time.time() - start_time
        tf.logging.info(
            "Average training steps per second: %.2f", self._training_steps / time_delta
        )

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info("Starting iteration %d", iteration)
        # pylint: disable=protected-access
        if not self._agent._replay_suffix:
            # Reload the replay buffer
            self._agent._replay.memory.reload_buffer(num_buffers=5)
        # pylint: enable=protected-access
        self._run_train_phase()

        # num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)

        # self._save_tensorboard_summaries(
        #     iteration, num_episodes_eval, average_reward_eval
        # )

        offline_evaluation(iteration)

        return statistics.data_lists

    # def _save_tensorboard_summaries(
    #     self, iteration, num_episodes_eval, average_reward_eval
    # ):
    #     """Save statistics as tensorboard summaries.

    #     Args:
    #       iteration: int, The current iteration number.
    #       num_episodes_eval: int, number of evaluation episodes run.
    #       average_reward_eval: float, The average evaluation reward.
    #     """
    #     summary = tf.Summary(
    #         value=[
    #             tf.Summary.Value(
    #                 tag="Eval/NumEpisodes", simple_value=num_episodes_eval
    #             ),
    #             tf.Summary.Value(
    #                 tag="Eval/AverageReturns", simple_value=average_reward_eval
    #             ),
    #         ]
    #     )
    #     self._summary_writer.add_summary(summary, iteration)

    def set_offline_evaluation(self, dataset_path, chkpt_path):
        # Get validation dataset
        self.dataset_path = dataset_path
        input_size = self._environment.observation_shape[0]
        self.predictor = RewardPredictor(input_size, os.path.abspath(chkpt_path))

    def offline_evaluation(self, iteration):
        self._agent.eval_mode = True

        validation_dataset = [
            os.path.join(self.dataset_path, f)
            for f in os.listdir(self.dataset_path)
            if os.path.isfile(os.path.join(self.dataset_path, f))
        ]

        actions = []
        estimation_eps = {
            "dm/score": [],
            "dm/pred_reward_mean": [],
            "dm/pred_reward_total": [],
        }
        for n_eps in range(len(validation_dataset[0])):
            reader = JsonReader(validation_dataset[0])
            batch = reader.next()
            estimation = {
                "dm/score": [],
                "dm/pred_reward_mean": [],
                "dm/pred_reward_total": [],
            }
            for episode in batch.split_by_episode():
                action = []
                action_probs = []
                for i in range(len(episode["eps_id"])):
                    action.append(
                        self._agent.step(episode["rewards"][i], episode["obs"][i])
                    )
                    action_probs.append(1.0)

                actions.extend(action)
                action = np.array([action])
                action_probs = np.array([action_probs])

                obs = torch.Tensor(
                    np.concatenate(
                        (episode["obs"], np.reshape(action, (action[0].shape[0], 1))),
                        axis=1,
                    )
                )  # concatenate actions and observations for input obs are usually [[obs1],[obs2],[obs3]] and
                # actions are usually [1,0,1,0] so the goal is to make actions like this: [[1],[0],[1]]
                scores_raw = self.predictor.predict(obs).detach().numpy()
                scores = {}
                scores["score"] = (scores_raw * action_probs).mean()
                scores["pred_reward_mean"] = scores_raw.mean()
                scores["pred_reward_total"] = scores_raw.sum()

                estimation["dm/score"].append(scores["score"])
                estimation["dm/pred_reward_mean"].append(scores["pred_reward_mean"])
                estimation["dm/pred_reward_total"].append(scores["pred_reward_total"])

            estimation_eps["dm/score"].append(mean(estimation["dm/score"]))
            estimation_eps["dm/pred_reward_mean"].append(
                mean(estimation["dm/pred_reward_mean"])
            )
            estimation_eps["dm/pred_reward_total"].append(
                mean(estimation["dm/pred_reward_total"])
            )

        est_mean = pd.DataFrame.from_dict(estimation_eps).mean(axis=0)

        summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag="Eval/DM/score", simple_value=est_mean["dm/score"]
                ),
                tf.Summary.Value(
                    tag="Eval/DM/pred_reward_mean",
                    simple_value=est_mean["dm/pred_reward_mean"],
                ),
                tf.Summary.Value(
                    tag="Eval/DM/pred_reward_total",
                    simple_value=est_mean["dm/pred_reward_total"],
                ),
            ]
        )
        self._summary_writer.add_summary(summary, iteration)