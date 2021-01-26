DATASET_NAME="v5_scoremax_dataset_cql"


mkdir -p cql-dataset/${DATASET_NAME}/
gsutil -m rsync -r gs://ac-rl-artifacts/data/processed/cql-dataset/${DATASET_NAME} cql-dataset/${DATASET_NAME}

# For 1% data, use minq_weight=4.0 and for 10% data, use minq_weight=1.0.
python -m batch_rl.fixed_replay.train \
--gin-bindings "['FixedReplayRunner.num_iterations=1000', 'FixedReplayQuantileAgent.minq_weight=1.0']"