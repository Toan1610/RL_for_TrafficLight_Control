# Module MGMQ_v2: Lane-level GAT

## 1. Vai tro

Lane-level GAT tao intersection embedding tu du lieu lane-level (12 lane slots).

Muc tieu:

- Hoc quan he cooperation giua cac lane cung pha.
- Hoc quan he conflict giua cac lane xung dot.
- Tao embedding gon nhung giu thong tin quan trong cho policy.

## 2. Input

Input vao `DualStreamGATLayer`:

- `lane_features`: [B, 12, F]
  - F thuong = 4 (density, queue, occupancy, average_speed)
- `adj_coop`: [B, 12, 12] (cooperation)
- `adj_conf`: [B, 12, 12] (conflict)

Hai adjacency matrix static duoc tao tu:

- `get_lane_cooperation_matrix()`
- `get_lane_conflict_matrix()`

## 3. Co che xu ly

`DualStreamGATLayer` thuc hien:

1. Input projection (`in_features -> hidden_dim`).
2. GAT stream cho coop graph.
3. GAT stream cho conflict graph.
4. Concat `[h, h_same, h_diff]`.
5. Final projection + activation.
6. Residual connection + LayerNorm.

## 4. Output

- `gat_out`: [B, 12, gat_output_dim * num_heads]
- Sau do encoder dung mean pooling theo lane:
  - `intersection_emb`: [B, gat_output_dim * num_heads]

Mean pooling giup giu kich thuoc embedding nho va on dinh hon flatten.

## 5. Rang buoc va kiem tra

- Observation da duoc clip [0,1] truoc khi vao model.
- Feature order bat buoc lane-major de reshape dung theo 12 lanes.
- Masking pha khong nam trong GAT; masking duoc xu ly o action distribution.
