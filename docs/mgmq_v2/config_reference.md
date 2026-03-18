# MGMQ_v2 Config Reference

Tai lieu nay tong hop nhom tham so trong `src/config/model_config.yml`.

## 1. `network`

- `name`: ten mang (`PhuQuoc`, ...)
- `base_path`: null -> mac dinh `network/{name}`
- `net_file`, `route_files`, `detector_file`, `intersection_config`

## 2. `mgmq`

- `gat.hidden_dim`, `gat.output_dim`, `gat.num_heads`
- `graphsage.hidden_dim`
- `gru.hidden_dim`
- `policy.hidden_dims`, `value.hidden_dims`
- `dropout`
- `history_length`
- `vf_share_coeff`
- `local_gnn.enabled`, `local_gnn.max_neighbors`, `local_gnn.obs_dim`

Gia tri hien tai trong file:

- GAT: 32 / 16 / 2
- GraphSAGE hidden: 64
- GRU hidden: 32
- Policy/Value MLP: [128, 64]
- history_length: 6
- vf_share_coeff: 1.0

## 3. `ppo`

- `learning_rate`
- `gamma`, `lambda_`
- `clip_param`
- `entropy_coeff`
- `kl_coeff`, `kl_target`
- `vf_clip_param`, `vf_loss_coeff`
- `train_batch_size`, `minibatch_size`, `num_sgd_iter`
- `grad_clip`

Gia tri hien tai:

- `clip_param: 0.15`

Nhan xet:

- 0.15 la muc an toan cho batch vua/nho, giam nguy co update qua manh.
- Neu hoc qua cham co the thu 0.2.
- Neu KL dao dong manh, co the giam 0.1.

## 4. `training`

- `num_iterations`, `num_workers`, `num_envs_per_worker`
- `checkpoint_interval`, `patience`
- `seed`, `use_gpu`, `output_dir`

## 5. `reward`

- `functions`: list reward names
- `weights`: trong so tuong ung

Validation quan trong:

- `len(weights) == len(functions)` khi co scalarization.

## 6. `action`

- `mode`: `discrete_adjustment` (khuyen nghi) hoac `ratio`
- `green_time_step`: buoc dieu chinh giay xanh moi pha

## 7. `environment`

- `num_seconds`
- `cycle_time`, `yellow_time`
- `min_green`, `max_green`
- `time_to_teleport`
- `use_phase_standardizer`
- `normalize_reward`, `clip_rewards`

Luu y runtime:

- Trong `SumoEnvironment`, `delta_time` luon bi ep = `cycle_time`.
- Trong RLlib train path, `normalize_reward=true` kich hoat GESA reward wrapper.

## 8. Checklist consistency

- File network/route/detector ton tai.
- `intersection_config.json` dung voi mang dang chay.
- `cycle_time > yellow_time`.
- `max_green > min_green`.
- Reward function names nam trong list da register.
