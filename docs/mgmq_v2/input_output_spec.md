# MGMQ_v2 Input / Output Specification

## 1. Input he thong

### 1.1 Input files

- Network: `*.net.xml`
- Route(s): `*.rou.xml` (co the nhieu file)
- Detector: `detector.add.xml`
- Preprocessing: `intersection_config.json`
- Config: `src/config/model_config.yml`

### 1.2 Input env config

Cac nhom quan trong:

- `network`: ten mang + duong dan file
- `reward`: functions + weights
- `action`: mode + green_time_step
- `environment`: cycle_time, min/max_green, teleport, normalize_reward

## 2. Observation input cho model

## 2.1 Mac dinh (global MGMQ)

Observation dict:

```text
features: float32[48]
action_mask: float32[8]
```

`features` = 12 lane slots x 4 metrics:

- density
- queue
- occupancy
- average_speed

Tat ca duoc clip [0,1].

## 2.2 Temporal mode

Neu dung `SpatioTemporalObservationFunction`:

- `features`: float32[window_size * 48]
- `action_mask`: float32[8]

## 2.3 Local GNN mode

Neu `local_gnn.enabled=true`:

```text
self_features: float32[T, F]
neighbor_features: float32[K, T, F]
neighbor_mask: float32[K]
neighbor_directions: float32[K]
action_mask: float32[8]
```

Trong do thuong `F=48`, `K=max_neighbors`.

## 3. Action output tu policy

## 3.1 Discrete mode (mac dinh)

- Policy logits: float[B, 24]
- Sample action: int[8], moi phan tu thuoc {0,1,2}

Mapping:

- 0 -> decrease
- 1 -> keep
- 2 -> increase

## 3.2 Ratio mode (legacy)

- Action continuous float[8].

## 4. Output tu env/simulator

Sau khi dich action:

- `green_times`: list[int] theo pha THUC TE cua nut giao.
- Tong green giu bang `total_green_time` trong chu ky.
- `next_action_time = current_time + cycle_time`.

## 5. Output training

- Checkpoints (Ray Tune / RLlib)
- `mgmq_training_config.json`
- `normalizer_state.json` (neu reward normalization bat)
- Progress logs

## 6. Output evaluation

JSON metrics thuong gom:

- `mean_raw_reward`, `std_raw_reward`
- `mean_reward`, `std_reward`
- `episode_lengths`
- `mean_waiting_time`, `mean_avg_speed`, `mean_total_halts`, `mean_throughput`, `mean_pressure`
- `per_agent_stats`

## 7. I/O contracts can dung

- Shape obs phai on dinh theo mode model.
- `action_mask` phai dong bo voi mapping FRAP.
- `cycle_time`, `min_green`, `max_green`, `yellow_time` phai tao duoc green split kha thi.
