#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot cumulative measure vs arrival time from SUMO tripinfo XML files.

Example:
  python plot_cumulative_measure.py \
    --input atsc.xml,fixed.xml \
    --label "ATSC,FIXED" \
    --measure waitingTime \
    --title "Tổng thời gian chờ" \
    --out total_waiting.png
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
import math

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Vẽ Y = tổng (cumulative) của một thuộc tính trong tripinfo theo X = arrival time."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Danh sách file tripinfo .xml, phân tách bằng dấu phẩy. VD: a.xml,b.xml",
    )
    p.add_argument(
        "--label",
        required=True,
        help='Danh sách nhãn cho các đường, phân tách bằng dấu phẩy. VD: "ATSC,FIXED"',
    )
    p.add_argument(
        "--measure",
        required=True,
        help="Tên thuộc tính đo trong <tripinfo>, ví dụ: waitingTime, timeLoss, duration...",
    )
    p.add_argument(
        "--title",
        required=True,
        help="Tiêu đề biểu đồ.",
    )
    p.add_argument(
        "--out",
        default="",
        help="(Tuỳ chọn) Đường dẫn file ảnh để lưu (png/jpg/svg). Nếu bỏ trống sẽ chỉ hiển thị.",
    )
    p.add_argument(
        "--min-arrival",
        type=float,
        default=None,
        help="(Tuỳ chọn) Chỉ vẽ các chuyến có arrival >= giá trị này.",
    )
    p.add_argument(
        "--max-arrival",
        type=float,
        default=None,
        help="(Tuỳ chọn) Chỉ vẽ các chuyến có arrival <= giá trị này.",
    )
    p.add_argument(
        "--skip-nan",
        action="store_true",
        help="Bỏ qua (đếm = 0) nếu bản ghi không có measure hoặc không chuyển được sang số.",
    )
    p.add_argument(
        "--marker-every",
        type=int,
        default=0,
        help="(Tuỳ chọn) Vẽ marker mỗi N điểm (0 = không vẽ marker).",
    )
    return p.parse_args()


def read_tripinfo_points(path: Path, measure: str) -> List[Tuple[float, float]]:
    """
    Đọc một file tripinfo XML, trả về list (arrival, measure_value).
    - arrival: float từ thuộc tính 'arrival'
    - measure_value: float từ thuộc tính 'measure' yêu cầu
    """
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise RuntimeError(f"Lỗi parse XML {path}: {e}") from e

    points: List[Tuple[float, float]] = []
    # Tương thích cả khi root không phải <tripinfos> (một số phiên bản wrap khác)
    for elem in root.iter("tripinfo"):
        arr_s = elem.get("arrival")
        val_s = elem.get(measure)
        if arr_s is None:
            # Không có arrival -> bỏ qua
            continue

        try:
            arrival = float(arr_s)
        except (TypeError, ValueError):
            # arrival không hợp lệ
            continue

        if val_s is None:
            # Không có measure
            value = math.nan
        else:
            try:
                value = float(val_s)
            except (TypeError, ValueError):
                value = math.nan

        points.append((arrival, value))

    return points


def filter_and_sort(points: List[Tuple[float, float]],
                    min_arrival: float | None = None,
                    max_arrival: float | None = None) -> List[Tuple[float, float]]:
    out = []
    for t, v in points:
        if min_arrival is not None and t < min_arrival:
            continue
        if max_arrival is not None and t > max_arrival:
            continue
        out.append((t, v))
    # Sắp theo arrival tăng dần (ổn định)
    out.sort(key=lambda x: x[0])
    return out


def cumulative_series(points: List[Tuple[float, float]], skip_nan: bool) -> Tuple[list, list]:
    """
    Tạo chuỗi (X=arrival_list, Y=cumulative_sum_list).
    - Nếu skip_nan=True: nan -> coi như 0 để cộng dồn.
    - Nếu skip_nan=False: gặp nan -> bỏ hẳn điểm đó (không thêm vào chuỗi).
    """
    xs, ys = [], []
    running = 0.0
    for t, v in points:
        if v is None or math.isnan(v):
            if skip_nan:
                v = 0.0
            else:
                # Bỏ điểm
                continue
        running += v
        xs.append(t)
        ys.append(running)
    return xs, ys


def main():
    args = parse_args()
    files = [Path(s.strip()) for s in args.input.split(",") if s.strip()]
    labels = [s.strip() for s in args.label.split(",") if s.strip()]

    if len(files) == 0:
        print("⚠️  Chưa cung cấp file trong --input", file=sys.stderr)
        sys.exit(2)
    if len(labels) != len(files):
        print("⚠️  Số lượng label phải bằng số file (--label)", file=sys.stderr)
        sys.exit(2)

    plt.figure(figsize=(9, 5.5))

    plotted_any = False
    for path, lab in zip(files, labels):
        pts = read_tripinfo_points(path, args.measure)
        if len(pts) == 0:
            print(f"⚠️  {path}: không tìm thấy <tripinfo> hợp lệ. Bỏ qua.", file=sys.stderr)
            continue

        pts = filter_and_sort(pts, args.min_arrival, args.max_arrival)
        if len(pts) == 0:
            print(f"⚠️  {path}: không còn điểm sau lọc arrival. Bỏ qua.", file=sys.stderr)
            continue

        xs, ys = cumulative_series(pts, skip_nan=args.skip_nan)
        if len(xs) == 0:
            print(f"⚠️  {path}: tất cả điểm bị loại (nan và --skip-nan không bật). Bỏ qua.", file=sys.stderr)
            continue

        if args.marker_every and args.marker_every > 0:
            plt.plot(xs, ys, label=lab, marker="o", markevery=args.marker_every)
        else:
            plt.plot(xs, ys, label=lab)

        plotted_any = True

    if not plotted_any:
        print("❌ Không có đường nào để vẽ. Kiểm tra input/measure.", file=sys.stderr)
        sys.exit(3)

    plt.xlabel("Arrival time (s)")
    plt.ylabel(f"Tổng {args.measure}")
    plt.title(args.title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"✅ Đã lưu biểu đồ: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
