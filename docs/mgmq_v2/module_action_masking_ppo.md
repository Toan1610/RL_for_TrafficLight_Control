# Module MGMQ_v2: Action Masking + PPO Policy

## 1. Kien truc policy hien tai

MGMQ_v2 dung PPO (`MGMQPPOTorchPolicy`), khong dung DDQN.

Model wrapper (`MGMQTorchModel` / `LocalMGMQTorchModel`) output:

- Policy logits
- Value prediction

## 2. Action space

### Mac dinh: `discrete_adjustment`

- Space: `MultiDiscrete([3] * 8)`
- Moi standard phase co 3 hanh dong:
  - `0`: decrease (`-green_time_step`)
  - `1`: keep (`0`)
  - `2`: increase (`+green_time_step`)

### Legacy: `ratio`

- Space: `Box[8]`
- Dung masked softmax distribution.

## 3. Action masking trong discrete mode

Action distribution: `masked_multi_categorical`.

- Input logits: [B, 24] -> reshape [B, 8, 3].
- Lay `action_mask` tu model (`_last_action_mask`).
- Phase invalid (mask=0) bi ep logits thanh `[-inf, 0, -inf]`.

He qua:

- Phase invalid luon chon `keep`.
- Khong lang phi gradient vao action vo hieu.

## 4. PPO loss va diagnostics

`MGMQPPOTorchPolicy.loss()` gom:

- PPO clipped surrogate loss
- Value loss (`vf_loss_coeff`, `vf_clip_param`)
- Entropy bonus (`entropy_coeff`)
- KL penalty (`kl_coeff`, `kl_target`)

Custom bo sung trong code hien tai:

1. Per-minibatch advantage normalization (mean=0, std=1).
2. `clip_fraction` tracking.
3. Advantage raw stats (`adv_std_raw`, `adv_max_abs_raw`, ...).
4. Gradient cosine similarity giua policy loss va value loss tren shared encoder.

## 5. Lien ket voi cycle control

Sau khi sample action:

- Action duoc dich sang green split cho CHU KY tiep theo.
- Khong switch pha tung giay.

Cach nay phu hop voi luong den dem lui va giu chu ky on dinh.
