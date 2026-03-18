# Luong Thuat toan MGMQ_v2 (End-to-End)

Tai lieu nay mo ta luong thuc thi tu input SUMO den training/eval theo code hien tai.

## 1. Input he thong

- Network: `.net.xml`
- Routes: 1 hoac nhieu `.rou.xml`
- Detector: `detector.add.xml`
- Preprocess config: `intersection_config.json`
- Runtime config: `src/config/model_config.yml`

## 2. Buoc 0 - Preprocessing offline

Script `scripts/preprocess_network.py` sinh/cap nhat `intersection_config.json`:

- `intersections[ts_id].direction_map`
- `intersections[ts_id].phase_config.actual_to_standard`
- `intersections[ts_id].phase_config.standard_to_actual`
- `adjacency`

Muc dich: chuan hoa topology de model dung chung tren nhieu giao lo.

## 3. Buoc 1 - Tao env + simulator

1. Doc YAML config (`network`, `reward`, `action`, `environment`, ...).
2. Resolve path file network/route/detector/intersection config.
3. Tao `SumoEnvironment` -> `SumoSimulator` -> `TrafficSignal` cho tung TS co E2 detector.
4. Neu RLlib path: env duoc wrap boi GESA wrapper qua `register_sumo_env`.

Quy tac quan trong:

- `delta_time` luon = `cycle_time`.
- 1 decision / cycle / intersection.

## 4. Buoc 2 - Observation moi step

Mac dinh (`DefaultObservationFunction`):

```text
{
  "features": float32[48],
  "action_mask": float32[8]
}
```

`features` duoc xep lane-major: moi lane gom 4 so `[density, queue, occupancy, average_speed]`, sau do clip [0,1].

## 5. Buoc 3 - Forward qua MGMQ model

1. Tach `action_mask` va `features` tu `obs_flat`.
2. Encoder:
   - `DualStreamGATLayer` tren 12 lanes.
   - Mean pooling lane embedding.
   - GraphSAGE + BiGRU de tong hop spatial context.
3. Policy head output logits.
4. Value head output `V(s)`.

Ghi chu implementation:

- Trong flatten order hien tai, model doc `action_mask` o dau vector flatten.

## 6. Buoc 4 - Action masking + sample action

Mode mac dinh: `discrete_adjustment`.

- Action space: `MultiDiscrete([3] * 8)`.
- Logits shape: `[B, 24]` = 8 phases x 3 actions.
- `masked_multi_categorical` xu ly mask:
  - phase invalid -> logits ep thanh `[-inf, 0, -inf]`
  - tuc la phase do luon chon action `keep`.

## 7. Buoc 5 - Dich action sang green split thuc te

Tai `TrafficSignal._apply_discrete_cycle_adjustment`:

1. Map action `{0,1,2}` -> `{-step, 0, +step}` (giay).
2. Nhan voi `action_mask` de bo phase invalid.
3. Dung FRAP map 8 standard phases -> actual phases.
4. Cong vao `current_green_times`.
5. Clip theo `min_green`, `max_green`.
6. Rescale de giu tong green = `total_green_time`.
7. Lam tron so nguyen + phan bo remainder.

Sau do simulator set program logic va reset phase ve 0 de cycle moi bat dau dong bo.

## 8. Buoc 6 - Rollout trong chu ky

Simulator loop den khi co TS `time_to_act` (tuc den `next_action_time`):

- `simulationStep()` tung giay.
- cap nhat detector history.
- cap nhat departed/teleport trackers.

Den moc cycle tiep theo:

- tra observation moi
- tinh reward cho moi agent
- done khi `sim_step >= sim_max_time`

## 9. Buoc 7 - Reward va normalization

- Reward tinh tai `TrafficSignal.compute_reward()`:
  - 1 ham reward -> scalar
  - nhieu ham reward -> vector, co the scalarize bang `np.dot(reward, weights)`

Normalization:

- RLlib train path: neu `normalize_reward=true`, wrapper `GESARewardWrapper` normalize reward.
- Eval RL va baseline: script dat `normalize_reward=false` de bao cao raw reward cong bang.

## 10. Buoc 8 - PPO update

`MGMQPPOTorchPolicy` toi uu:

- clipped surrogate objective
- vf loss + vf clipping
- entropy bonus
- KL penalty

Bo sung custom:

- per-minibatch advantage normalization
- clip_fraction metric
- grad cosine metric (policy vs value tren shared encoder)

## 11. Output

- Checkpoint PPO
- `mgmq_training_config.json`
- `normalizer_state.json` (neu co reward normalization)
- Tune logs/progress
- Eval JSON (`mean_raw_reward`, waiting/speed/halt/throughput/pressure, per-agent)
