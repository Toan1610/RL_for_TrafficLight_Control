# Phân Tích Kỹ Thuật: Proximal Policy Optimization (PPO) trong MGMQ

Tài liệu này mô tả chi tiết về thuật toán **Proximal Policy Optimization (PPO)** được áp dụng trong dự án điều khiển đèn tín hiệu giao thông (ITS) sử dụng kiến trúc MGMQ (Multi-Graph Masking Q-Network approach adapted for Policy Gradient).

---

## 1. Tổng Quan

PPO là một thuật toán *On-policy Gradient* tìm cách cân bằng giữa:
1. **Ease of implementation:** Dễ cài đặt hơn TRPO.
2. **Sample efficiency:** Tận dụng dữ liệu tốt hơn.
3. **Ease of tuning:** Ít hyperparameter nhạy cảm hơn so với các thuật toán khác.

Trong dự án này, PPO được sử dụng để tối ưu hóa policy $\pi_\theta(a_t|s_t)$ nhằm điều khiển pha đèn giao thông.

> [!IMPORTANT]
> **v2.0.0+**: Dự án sử dụng **MGMQ-PPO** — một custom PPO algorithm với:
> - **Masked Softmax Distribution** (thay Dirichlet) cho action space
> - **Per-minibatch advantage normalization** thay vì normalize toàn bộ batch
> - **clip_fraction tracking** để monitor chất lượng clipping
> - **DiagnosticCallback** log entropy, KL, gradient norms mỗi iteration

---

## 2. Các Biến Thể Chính của PPO

Thuật toán PPO có hai biến thể phổ biến nhất:

### 2.1 PPO-Clip (Clipped Surrogate Objective)
- Đây là phiên bản được sử dụng trong dự án này và cũng là mặc định của RLlib, Stable-Baselines3, CleanRL...
- Sử dụng hàm loss với cơ chế "clipping" để giới hạn mức độ thay đổi của policy trong mỗi lần cập nhật.
- **Ưu điểm**: Đơn giản, hiệu quả, dễ tune hyperparameter.

### 2.2 PPO-Penalty (Adaptive KL Penalty)
- Thay vì clipping, PPO-Penalty thêm một thành phần penalty vào loss dựa trên độ lệch KL-divergence giữa policy mới và cũ:
  $$L^{Penalty}(\theta) = \mathbb{E}_t [r_t(\theta) \hat{A}_t - \beta \cdot KL[\pi_{old}, \pi_\theta]]$$
- $\beta$ là hệ số penalty, có thể được điều chỉnh động dựa trên mức độ KL-divergence thực tế.
- **Ưu điểm**: Kiểm soát chính xác hơn mức độ thay đổi của policy, nhưng khó tune.

### 2.3 So sánh nhanh
| Biến thể | Cơ chế ổn định | Dễ tune | Được dùng phổ biến |
|----------|---------------|---------|-------------------|
| PPO-Clip | Clipping      | Dễ      | ⭐⭐⭐⭐⭐             |
| PPO-Penalty | KL Penalty  | Khó     | ⭐                 |

**Kết luận:**
- PPO-Clip là lựa chọn mặc định cho hầu hết các framework RL hiện đại.
- PPO-Penalty chỉ dùng khi cần kiểm soát cực kỳ chặt chẽ về độ thay đổi policy.

---

## 3. PPO Objective Function (Hàm Mục Tiêu)

Hàm loss tổng quát của PPO trong project được tính như sau:

$$L_t(\theta) = \hat{\mathbb{E}}_t [ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) ]$$

Trong đó:
* $L_t^{CLIP}$: Policy Loss (tối ưu hành động).
* $L_t^{VF}$: Value Function Loss (tối ưu dự đoán phần thưởng).
* $S$: Entropy Bonus (khuyến khích khám phá).
* $c_1, c_2$: Các hệ số trọng số (`vf_loss_coeff`, `entropy_coeff`).

### 3.1. Clipped Surrogate Objective ($L^{CLIP}$)
Đây là thành phần cốt lõi giúp PPO hoạt động ổn định, ngăn chặn việc cập nhật policy quá mạnh làm "hỏng" những gì model đã học.

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

* **Ratio $r_t(\theta)$**: Tỷ lệ xác suất hành động mới so với cũ.
    $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
* **Advantage $\hat{A}_t$**: Lợi thế của hành động $a_t$ so với mức trung bình. Được tính bằng **GAE (Generalized Advantage Estimation)**.
* **Clipping $\epsilon$**: Giới hạn thay đổi. Trong config hiện tại: `clip_param: 0.2`.
    * Nghĩa là policy mới không được lệch quá 20% so với policy cũ trong một bước update đơn lẻ.

### 3.2. Value Function Loss ($L^{VF}$)
Để tính Advantage, ta cần một Value Function $V(s)$ ước lượng tổng phần thưởng tích lũy.

$$L^{VF} = (V_\theta(s_t) - V_t^{target})^2$$

* Config hiện tại: `vf_clip_param: 10.0`. Điều này giúp cắt bớt các giá trị loss quá lớn do nhiễu, tránh gradient bùng nổ.

### 3.3. Entropy Bonus ($S$)
Entropy đo lường độ ngẫu nhiên của policy.
* **Công thức:** $S = -\sum \pi(a|s) \log \pi(a|s)$
* **Mục đích:** Ngăn policy hội tụ quá sớm vào một phương án cục bộ (sub-optimal).
* Config hiện tại: `entropy_coeff: 0.02`.

---

## 4. Masked Softmax Distribution cho Action Space (v2.0.0+)

Từ v2.0.0, dự án chuyển từ **Dirichlet Distribution** sang **Masked Softmax Distribution**.

### 4.1 Tại sao chuyển sang Masked Softmax?

| Vấn đề với Dirichlet | Giải pháp với Masked Softmax |
|-----------------------|------------------------------|
| Pha bị mask vẫn nhận xác suất > 0 (xấp xỉ) | Mask TRƯỚC softmax → xác suất = 0.0 **chính xác** |
| Gradient lãng phí cho pha invalid | Gradient chỉ chảy qua pha hợp lệ |
| Concentration param khó control entropy | `log_std` trực tiếp control exploration |
| Sample ngay simplex, nhưng không mask được | softmax tự động đảm bảo sum = 1.0 |

### 4.2 Cách hoạt động

```python
# Trong masked_softmax_distribution.py
class MaskedSoftmaxDistribution:
    def __init__(self, logits, log_std, action_mask, temperature=0.3):
        # logits: [batch, 8]  (8 pha chuẩn)
        # log_std: [batch, 8] (clamped [-5.0, 0.5])
        # action_mask: [batch, 8] (True = hợp lệ)
        
        std = torch.exp(log_std.clamp(-5.0, 0.5))
        noise = torch.randn_like(logits) * std
        logits_noisy = logits + noise
        
        # Mask BEFORE softmax: gán -1e9 cho pha invalid
        logits_noisy[~action_mask] = -1e9
        
        # Softmax với temperature
        self.probs = F.softmax(logits_noisy / temperature, dim=-1)
    
    def sample(self):
        return self.probs  # Đã là ratio hợp lệ (sum=1, invalid=0)
    
    def entropy(self):
        # Gaussian entropy: 0.5 * log(2πe * σ²) cho K-1 bậc tự do
        # Có thể âm khi log_std rất nhỏ (đã hội tụ)
        K = action_mask.sum(-1)  # số pha hợp lệ
        return 0.5 * (K - 1) * (1 + math.log(2 * math.pi)) + log_std[action_mask].sum()
```

### 4.3 Temperature và Entropy

| Tham số | Giá trị | Vai trò |
|---------|---------|--------|
| `temperature` | 0.3 | Control sharpness của softmax. Nhỏ hơn → sharp hơn |
| `log_std_min` | -5.0 | Giới hạn dưới của std (std ≈ 0.007 → rất deterministic) |
| `log_std_max` | 0.5 | Giới hạn trên của std (std ≈ 1.65 → exploration cao) |

> [!WARNING]
> **Entropy âm**: Với Masked Softmax, entropy sử dụng công thức Gaussian:
> $H = 0.5(K-1)\log(2\pi e) + \sum \log(\sigma_k)$
> Khi `log_std` rất nhỏ (< -1.42), entropy sẽ âm. Điều này bình thường và chỉ ra policy đã hội tụ.

---

## 5. Cấu Hình Hyperparameters Thực Tế

Dựa trên file `model_config.yml`, đây là cấu hình đang chạy:

| Tham số | Giá trị | Ý nghĩa | Phân tích |
|---------|---------|---------|-----------|
| `gamma` | 0.99 | Discount Factor | Ưu tiên phần thưởng dài hạn. Phù hợp cho giao thông vì hành động hiện tại ảnh hưởng lâu dài. |
| `lambda_` | 0.95 | GAE Parameter | Cân bằng giữa Bias và Variance khi tính Advantage. 0.95 là giá trị tiêu chuẩn. |
| `learning_rate` | 5e-4 | Learning Rate | Giảm từ 8e-4 → 5e-4 (v2.0.0+) cho ổn định. |
| `clip_param` | 0.2 | PPO Clip | Giới hạn thay đổi policy 20% mỗi update. |
| `vf_clip_param` | 10.0 | VF Clip | Clip value function loss. |
| `minibatch_size` | 64 | Batch Size cho SGD | Kích thước mẫu dùng để tính gradient. |
| `num_sgd_iter` | 10 | Epochs per Iteration | Số lần model học đi học lại trên cùng một batch. |
| `entropy_coeff` | 0.02 | Trọng số Entropy | Khuyến khích exploration. Có thể giảm khi model đã hội tụ. |
| `grad_clip` | 0.5 | Gradient Clipping | Ngăn gradient explosion. |
| `kl_target` | 0.01 | KL Target (v2.0.0+) | Target cho adaptive KL coefficient. |
| `kl_coeff` | 0.05 | KL Init Coeff (v2.0.0+) | Khởi tạo adaptive KL penalty. |
| `vf_share_layers` | true | Shared Encoder | Policy/Value dùng chung encoder (với `vf_share_coeff=0.01`). |

---

## 6. Kiến Trúc Model PPO-MGMQ

PPO trong dự án này không dùng mạng nơ-ron thẳng (MLP) thông thường mà sử dụng kiến trúc đồ thị (GNN) để trích xuất đặc trưng không gian:

### 6.1 High-level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     MGMQ-PPO Model                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Input Layer                                              │
│  ─────────────                                              │
│  Observation Dict:                                           │
│    - self_features: [B, T, 48]                              │
│    - neighbor_features: [B, K, T, 48]                       │
│    - neighbor_mask: [B, K]                                  │
│                                                              │
│  2. Encoder (LocalTemporalMGMQEncoder)                      │
│  ─────────────────────────────────────                      │
│    ├─→ DualStreamGATLayer (lane-level attention)            │
│    │     - Cooperation stream (same-phase lanes)            │
│    │     - Conflict stream (different-phase lanes)          │
│    │     - Multi-head attention (4 heads)                   │
│    │                                                        │
│    ├─→ DirectionalGraphSAGE (spatial aggregation)           │
│    │     - 5-direction projection (Self, N, E, S, W)        │
│    │     - Topology-aware neighbor pairing                  │
│    │                                                        │
│    └─→ TemporalGraphSAGE_BiGRU (temporal processing)        │
│          - Bi-directional GRU                               │
│          - Final network embedding                          │
│                                                              │
│  3. Joint Embedding                                          │
│  ─────────────────                                          │
│    concat(intersection_emb, network_emb) → [B, joint_dim]   │
│                                                              │
│  4. Policy Head (Actor)                                      │
│  ────────────────────                                       │
│    MLP [128, 64] → [logits, log_std] [8 pha chuẩn]         │
│    MaskedSoftmax(logits, log_std, mask, T=0.3) → ratio      │
│                                                              │
│  5. Value Head (Critic)                                      │
│  ─────────────────────                                      │
│    MLP [128, 64] → V(s) scalar                              │
│    (Shared encoder với vf_share_coeff=0.01)                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Model Outputs

| Output | Shape | Mô tả |
|--------|-------|-------|
| `action_dist` | MaskedSoftmax(B, 8) | Distribution over 8 standard phase time ratios |
| `action_sample` | [B, 8] | Sampled action (sum = 1.0, invalid = 0.0) |
| `action_log_prob` | [B] | Log probability of action |
| `value` | [B] | State value estimate |

### 6.3 Log-std Bounds

```python
# Trong masked_softmax_distribution.py
SOFTMAX_LOG_STD_MIN = -5.0   # std = e^-5 ≈ 0.007 (very deterministic)
SOFTMAX_LOG_STD_MAX = 0.5    # std = e^0.5 ≈ 1.65 (high exploration)
```

### 6.4 MGMQ-PPO Custom Algorithm (v2.0.0+)

**Per-minibatch Advantage Normalization:**

Thay vì normalize advantage trên toàn bộ train batch (mặc định RLlib), MGMQ-PPO normalize **trong mỗi minibatch**:

```python
# Trong mgmq_ppo.py
class MGMQPPO(PPO):
    def training_step(self):
        # Override để thêm per-minibatch advantage normalization
        for minibatch in sample_minibatches(train_batch):
            adv = minibatch["advantages"]
            # Normalize trong minibatch thay vì toàn batch
            adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
            minibatch["advantages"] = adv_normalized
            
            # Track clip_fraction
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped = ((ratio - 1.0).abs() > clip_param).float().mean()
            metrics["clip_fraction"] = clipped.item()
```

**Lợi ích:**
- Giảm variance trong gradient estimation
- Minibatch nhỏ có advantage phân phối chuẩn hơn
- `clip_fraction` cho biết % samples bị clip → monitor policy stability

---

## 7. Các Metrics Quan Trọng Cần Theo Dõi

Khi training, cần quan sát các chỉ số sau trong Tensorboard hoặc console:

### 7.1 Primary Metrics

| Metric | Mục tiêu | Phân tích |
|--------|----------|-----------|
| `episode_reward_mean` | Tăng dần | Nếu không tăng → check reward function, learning rate |
| `episode_len_mean` | Ổn định | Nếu giảm → episodes ending early (bad policy) |
| `policy_loss` | Giảm rồi ổn định | Nếu không giảm → learning rate quá thấp |
| `vf_loss` | Giảm | Value function đang học |
| `entropy` | Giảm từ từ | Giảm quá nhanh → exploration không đủ |

### 7.2 PPO-specific Metrics

| Metric | Giá trị tốt | Cảnh báo |
|--------|-------------|----------|
| `kl` (KL Divergence) | 0.005 - 0.02 | > 0.05: policy changing too fast |
| `vf_explained_var` | > 0.5 | < 0.3: value function struggling |
| `approx_kl` | < `kl_target` | High: reduce learning rate |
| `clip_fraction` | 0.1 - 0.3 | > 0.5: clip_param too small |

### 7.3 Traffic-specific Metrics (Custom)

| Metric | Mô tả | Mục tiêu |
|--------|-------|----------|
| `total_waiting_time` | Tổng thời gian chờ | Giảm |
| `throughput` | Số xe đi qua/giờ | Tăng |
| `halting_vehicles` | Số xe đang dừng | Giảm |
| `average_speed` | Tốc độ TB | Tăng |

---

## 8. Training Loop Flow

```mermaid
graph TD
    A[Start Iteration] --> B[Collect Rollouts]
    B --> C[Compute Advantages<br/>using GAE]
    C --> D[Compute Loss<br/>Policy + Value + Entropy]
    D --> E{num_sgd_iter<br/>completed?}
    E -->|No| F[Update Model<br/>with minibatch]
    F --> E
    E -->|Yes| G[Log Metrics]
    G --> H{Converged?}
    H -->|No| A
    H -->|Yes| I[Save Checkpoint]
```

---

## 9. Kết Luận

Setup MGMQ-PPO hiện tại (v2.0.0+) đã được nâng cấp đáng kể:

1. **Masked Softmax Distribution** (thay Dirichlet): Masking TRƯỚC softmax → pha invalid nhận xác suất = 0.0 chính xác, gradient chỉ chảy qua pha hợp lệ.

2. **MGMQ-PPO Custom Algorithm**: Per-minibatch advantage normalization, clip_fraction tracking, tích hợp Masked Softmax.

3. **GESA Wrapper**: Obs running mean/std normalization + reward scaling → gradient ổn định, value function dễ học.

4. **DiagnosticCallback**: Log entropy, KL, gradient norms, advantage stats mỗi iteration → phát hiện sớm policy collapse.

5. **DualStreamGATLayer**: Xử lý cả cooperation (same-phase) và conflict (different-phase) relationships giữa các lanes.

6. **Directional GraphSAGE**: Bảo toàn thông tin hướng của dòng giao thông.

7. **Kết quả**: Checkpoint 32 đạt `mean_raw_reward = -41.17` vs Baseline `-52.96` → **+22.3% improvement**.

**Bước tiếp theo:**
- Monitor entropy để tránh policy collapse (entropy âm quá sâu)
- Tune `kl_target` nếu `kl_coeff` tăng quá nhanh
- Thử nghiệm trên mạng lưới lớn hơn (grid4x4, PhuQuoc)
