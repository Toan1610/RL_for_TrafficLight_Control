# MGMQ_v2 Documentation

Bo tai lieu nay mo ta luong MGMQ_v2 theo implementation hien tai trong project.

## Muc luc

1. [Tong quan kien truc](./architecture_overview.md)
2. [Luong thuat toan end-to-end](./algorithm_flow.md)
3. [Module lane-level GAT](./module_lane_level_gat.md)
4. [Module directional GraphSAGE + BiGRU](./module_directional_graphsage.md)
5. [Module action masking + PPO policy](./module_action_masking_ppo.md)
6. [Reward functions](./reward_functions.md)
7. [Dac ta dau vao / dau ra](./input_output_spec.md)
8. [Bang tham so cau hinh](./config_reference.md)
9. [Luong train/eval/benchmark](./training_evaluation_flow.md)

## Pham vi MGMQ_v2

- Kien truc multi-agent cho dieu khien den giao thong theo chu ky.
- Action mode mac dinh: `discrete_adjustment` voi `MultiDiscrete([3] * 8)`.
- Policy learning: PPO (custom policy `MGMQPPOTorchPolicy`), khong con DDQN.
- Chuan hoa pha qua GPI + FRAP (`intersection_config.json`).
- Action masking ep pha khong hop le ve hanh dong `keep`.
- Reward co the la 1 ham hoac danh sach ham + weights.

## Tai lieu goc de doi chieu

- `docs/origin_docs/MGMQ_docs/`

Bo `mgmq_v2` giu cach chia file/module tuong tu bo cu, nhung noi dung la luong dang chay thuc te.
