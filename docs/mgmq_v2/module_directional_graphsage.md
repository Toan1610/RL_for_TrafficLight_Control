# Module MGMQ_v2: Directional GraphSAGE + BiGRU

## 1. Vai tro

Module nay tong hop spatial context giua giao lo hien tai va giao lo lan can.

Diem quan trong: BiGRU o day xu ly sequence theo KHONG GIAN (huong/neighbor), khong phai temporal RNN.

## 2. Global mode (mac dinh)

Khi dung `MGMQEncoder`:

- Input node features: intersection embedding tu lane-level GAT.
- Adjacency: directional matrix [4, N, N] (N,E,S,W).

`DirectionalGraphSAGE` lam:

1. Project input thanh 5 vectors: self, north, east, south, west.
2. Topology-aware exchange:
   - in_north nhan tu south cua neighbors
   - in_east nhan tu west cua neighbors
   - ...
3. Stack chuoi huong `[N, E, S, W]`.
4. BiGRU aggregate chuoi huong.
5. Concat voi self vector -> output projection.
6. Residual + LayerNorm.

Sau do MGMQ encoder mean-pool network embedding theo agents de lay context toan mang.

## 3. Local mode (tuy chon)

Khi `local_gnn.enabled=true`, dung `LocalMGMQEncoder` + `NeighborGraphSAGE_BiGRU`:

- Moi agent nhan neighbor features da dong goi san (`K` neighbors).
- Co `neighbor_mask` va `neighbor_directions`.
- Projection theo tung direction (N/E/S/W) + fallback projection.
- BiGRU aggregate theo thu tu neighbors (spatial).

Local mode tranh van de global graph bi vo cau truc khi RLlib shuffle minibatch.

## 4. Output

Output cua tang nay la network embedding, sau do concat voi intersection embedding:

- `joint_emb = [intersection_emb || network_emb]`

`joint_emb` duoc dua vao policy head va value head.
