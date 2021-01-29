mkdir -p cql-dataset/v5_scoremax_dataset_cql/
gsutil -m rsync -r gs://ac-rl-artifacts/data/processed/cql-dataset/v5_scoremax_dataset_cql cql-dataset/v5_scoremax_dataset_cql

mkdir -p data/processed/v5_dataset/test_dataset_users/
gsutil -m rsync -r gs://ac-rl-artifacts/data/processed/v5_dataset/test_dataset_users/ data/processed/v5_dataset/test_dataset_users/

mkdir -p models/reward_pred_v0_model/release/80_input/
gsutil -m rsync -r gs://ac-rl-artifacts/models/reward_pred_v0_model/release/80_input/ models/reward_pred_v0_model/release/80_input/

# For 1% data, use minq_weight=4.0 and for 10% data, use minq_weight=1.0.
python -m batch_rl.fixed_replay.train \
--gin-bindings "['FixedReplayRunner.num_iterations=1000', 'FixedReplayQuantileAgent.minq_weight=4.0']"