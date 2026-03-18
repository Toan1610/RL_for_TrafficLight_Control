# MGMQ_v2 Training / Evaluation / Baseline Flow

## 1. Pipeline tong quat

1. Preprocess network -> tao/cap nhat `intersection_config.json`.
2. Train PPO (`scripts/train_mgmq_ppo.py`).
3. Eval RL (`scripts/eval_mgmq_ppo.py`).
4. Eval baseline (`tools/eval_baseline_reward.py`).
5. So sanh bang `mean_raw_reward`.

## 2. Preprocess

```bash
python scripts/preprocess_network.py --network PhuQuoc
```

Neu can:

```bash
python scripts/preprocess_network.py --network PhuQuoc --detector-file network/PhuQuoc/detector.add.xml
```

## 3. Train RL

```bash
python scripts/train_mgmq_ppo.py --config src/config/model_config.yml
```

Output:

- checkpoints
- `mgmq_training_config.json`
- training logs

## 4. Eval RL

```bash
python scripts/eval_mgmq_ppo.py \
  --checkpoint <checkpoint_path> \
  --network PhuQuoc \
  --episodes 5 \
  --seeds 42 43 44 45 46
```

Script eval RL dat `normalize_reward=false` de bao cao raw reward cong bang voi baseline.

## 5. Eval baseline

### 5.1 Fixed-time

```bash
python tools/eval_baseline_reward.py \
  --controller fixed \
  --network PhuQuoc \
  --episodes 5 \
  --seeds 42 43 44 45 46
```

### 5.2 MaxPressure

```bash
python tools/eval_baseline_reward.py \
  --controller max_pressure \
  --network PhuQuoc \
  --episodes 5 \
  --seeds 42 43 44 45 46
```

Neu auto-discovery khong tim thay file `net-info.json`, truyen them:

```bash
--mp-net-info <path_to_net-info.json>
```

## 6. `--mp-net-info` co can cho RL eval khong?

- Khong. `--mp-net-info` chi dung cho baseline `max_pressure`.
- RL eval (`eval_mgmq_ppo.py`) khong su dung MaxPressure net-info.

## 7. Cach tinh muc tieu vuot baseline

So sanh bang raw reward:

```text
improvement_pct = (RL_raw - MP_raw) / abs(MP_raw) * 100
```

Muc tieu cua ban:

- `improvement_pct >= 20%` so voi MaxPressure.

## 8. Checklist fairness truoc khi ket luan

- Cung network file + route files.
- Cung seeds.
- Cung `num_seconds`, `cycle_time`, teleport setting.
- Cung reward config khi benchmark trong cung frame.
- So sanh bang `mean_raw_reward`, khong dung normalized reward.
