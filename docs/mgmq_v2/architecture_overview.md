# Tong quan Kien truc MGMQ_v2

## 1. Muc tieu thiet ke

MGMQ_v2 giai bai toan dieu khien den da giao lo voi 4 muc tieu:

- Giam waiting time theo chu ky.
- Giam ket xe (queue, halt, teleport).
- Tang throughput (departed vehicles).
- Dam bao hanh dong luon hop le tren moi topology giao lo.

## 2. Kien truc tong the

### Tang A - Chuan hoa giao lo (GPI + FRAP)

- `IntersectionStandardizer` (GPI) map incoming edge ve 4 huong chuan `N/E/S/W`.
- `PhaseStandardizer` (FRAP) phan tich pha thuc te va tao map:
  - `actual_to_standard`
  - `standard_to_actual`
- Tao `phase_mask` de biet standard phase nao hop le tren tung nut.

Luu y: nut co it pha (vi du 2 pha) thi `actual_to_standard` chi co 2 key la dung logic.

### Tang B - Observation

Observation mac dinh moi agent la Dict:

- `features`: float32[48] = 12 lanes x 4 metrics
  - density, queue, occupancy, average_speed
- `action_mask`: float32[8]

Du lieu duoc clip ve [0, 1].

### Tang C - Encoder MGMQ

- Lane-level: `DualStreamGATLayer`
  - stream cooperation (lane cung pha)
  - stream conflict (lane xung dot)
- Network-level:
  - Global mode: `GraphSAGE_BiGRU` voi directional adjacency [4, N, N]
  - Local mode: `NeighborGraphSAGE_BiGRU` voi pre-packaged neighbor obs
- Joint embedding = intersection embedding + network embedding.

### Tang D - Policy/Control (PPO)

- `MGMQTorchModel` output logits cho policy va value cho critic.
- Action dist:
  - `masked_multi_categorical` khi `discrete_adjustment`
  - `masked_softmax` khi `ratio` (legacy)
- PPO custom (`MGMQPPOTorchPolicy`) co:
  - per-minibatch advantage normalization
  - clip_fraction tracking
  - gradient cosine diagnostic (policy vs value)

## 3. Khac biet chinh so voi MGMQ goc

- MGMQ goc: DDQN-centric.
- MGMQ_v2 hien tai: PPO-centric + custom masked distributions.
- Hanh dong hien tai dieu chinh green split theo chu ky, khong chon phase index tung giay.

## 4. Quy tac cycle va countdown

- Trong env, `delta_time` bi ep bang `cycle_time`.
- Moi agent ra quyet dinh 1 lan moi chu ky.
- Action tao green split cho chu ky tiep theo.
- Simulator set program logic moi va reset den ve phase 0 de bat dau cycle moi.

Co che nay dung voi den dem lui: thoi gian xanh duoc an dinh ngay dau chu ky va giam dan trong cycle.
