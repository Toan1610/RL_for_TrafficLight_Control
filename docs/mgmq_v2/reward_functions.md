# MGMQ_v2 Reward Functions

Tai lieu nay mo ta reward theo implementation hien tai trong `TrafficSignal`.

## 1. Co che tong hop reward

Config ho tro:

```yaml
reward:
  functions: [r1, r2, r3]
  weights: [w1, w2, w3]
```

- Neu 1 ham: reward scalar truc tiep.
- Neu nhieu ham:
  - tinh tung thanh phan -> vector
  - neu co `weights` -> scalar hoa bang `np.dot(vector, weights)`.

Luu y:

- So luong `weights` phai bang so luong `functions`.
- Key dung la `weights` (khong phai `weight`).

## 2. Cac reward co san

### Waiting-based

- `diff-waiting-time`
  - Cong thuc: `(W_prev - W_now) / (max_veh * delta_time) * 3`
  - Clip [-3, 3]
- `cycle-diff-waiting-time`
  - Cong thuc: `W_prev - W_now`
  - Khong ep cung scale [-3,3]

### Congestion-based

- `halt-veh-by-detectors`
  - `-3 * min(1, halting / max_veh)`
  - Range [-3, 0]
- `queue`
  - `-(queued / max_veh) * 3`
  - Clip [-3, 3]
- `occupancy`
  - `- occupancy_mean * 3`
  - Range [-3, 0]
- `pressure`
  - pressure tu detector (`occupancy - speed`)
  - Reward = `-pressure * 3`, clip [-3, 3]
- `presslight-pressure`
  - `-|veh_in - veh_out|/max_veh * 3`
  - Range [-3, 0]
- `hybrid-waiting-pressure`
  - `cycle_diff_waiting + pressure_penalty`

### Throughput / failure-based

- `diff-departed-veh`
  - Thuong theo ty le xe roi khoi vung detector
  - Range [0, 3]
- `teleport-penalty`
  - Phat xe bi teleport
  - Range [-3, 0]

## 3. Goi y cho luong hien tai

Neu muc tieu la can bang waiting + anti-gridlock + throughput, bo reward dang dung la hop ly:

- `halt-veh-by-detectors`
- `diff-waiting-time`
- `diff-departed-veh`

`occupancy` co the dung, nhung thuong trung y nghia voi `halt/queue`.
Nen chi them occupancy khi can bo sung tin hieu detector cho kich ban ban tin occupancy on dinh.

## 4. Logging va fair comparison

- Training co the dung normalized reward (qua wrapper).
- Danh gia so sanh RL vs baseline nen dung `mean_raw_reward`.

