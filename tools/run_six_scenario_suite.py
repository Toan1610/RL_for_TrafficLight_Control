"""Run a standardized 6-scenario robustness suite for traffic-signal control.

This tool does two things:
1) Generate deterministic scenario route/additional files.
2) Evaluate fixed-time, MaxPressure, and optional RL checkpoint on all scenarios.

The suite maps to these tests:
- Test 1: Stability (low volume + balanced demand)
- Test 2: Saturation (near-capacity demand)
- Test 3: Spatial adaptation (skewed 85/15 demand)
- Test 4: Temporal adaptation (shock increase at 1/3 horizon)
- Test 5: Cyclic adaptation (sinusoidal demand)
- Test 6: Exception handling (temporary lane blockage)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Add project root to import path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_env_config, get_network_config, load_model_config
from scripts.eval_mgmq_ppo import evaluate_mgmq
from tools.eval_baseline_reward import evaluate_baseline


@dataclass
class DemandTemplate:
    """Template vehicle/trip record extracted from a SUMO route file."""

    element: ET.Element
    depart: float
    from_edge: str


@dataclass
class ScenarioSpec:
    """Scenario metadata for generation and evaluation."""

    scenario_id: str
    test_name: str
    description: str


def _safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _load_route_templates(route_file: Path) -> Tuple[Dict[str, str], List[ET.Element], List[DemandTemplate]]:
    tree = ET.parse(route_file)
    root = tree.getroot()

    root_attrib = dict(root.attrib)
    non_demand: List[ET.Element] = []
    templates: List[DemandTemplate] = []

    for child in root:
        if child.tag in {"trip", "vehicle"}:
            templates.append(
                DemandTemplate(
                    element=copy.deepcopy(child),
                    depart=_safe_float(child.get("depart"), 0.0),
                    from_edge=(child.get("from") or ""),
                )
            )
        else:
            non_demand.append(copy.deepcopy(child))

    templates.sort(key=lambda t: t.depart)
    return root_attrib, non_demand, templates


def _load_non_demand_elements(route_file: Path) -> List[ET.Element]:
    """Load non-demand entries (e.g., vType/vTypeDistribution/route) from a route file."""
    tree = ET.parse(route_file)
    root = tree.getroot()
    return [copy.deepcopy(child) for child in root if child.tag not in {"trip", "vehicle"}]


def _merge_non_demand_elements(
    primary: Sequence[ET.Element],
    extras: Sequence[ET.Element],
) -> List[ET.Element]:
    """Merge non-demand XML elements while avoiding duplicates by (tag, id)."""
    merged: List[ET.Element] = []
    seen_keys: set[Tuple[str, str]] = set()

    def _key(elem: ET.Element) -> Tuple[str, str]:
        elem_id = elem.get("id") or ET.tostring(elem, encoding="unicode")
        return (elem.tag, elem_id)

    for elem in list(primary) + list(extras):
        key = _key(elem)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(copy.deepcopy(elem))

    return merged


def _count_demand_entries(route_file: Path) -> int:
    tree = ET.parse(route_file)
    root = tree.getroot()
    return sum(1 for c in root if c.tag in {"trip", "vehicle"})


def _detect_trip_source(route_files: Sequence[Path]) -> Path:
    scored = []
    for rf in route_files:
        scored.append((_count_demand_entries(rf), rf))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        raise RuntimeError("Could not find a route file containing trip/vehicle demand entries")
    return scored[0][1]


def _clamp_depart(depart: float, duration_seconds: float) -> float:
    return max(0.0, min(duration_seconds - 0.01, depart))


def _rescale_depart_horizon(
    templates: Sequence[DemandTemplate],
    duration_seconds: float,
) -> Tuple[List[DemandTemplate], float]:
    """Rescale source depart times to match target simulation horizon.

    Returns:
        (rescaled_templates, source_max_depart)
    """
    source_max_depart = max((t.depart for t in templates), default=0.0)
    if source_max_depart <= 0.0:
        return list(templates), source_max_depart

    scale = (duration_seconds - 0.01) / source_max_depart
    rescaled = [
        DemandTemplate(
            element=tpl.element,
            depart=_clamp_depart(tpl.depart * scale, duration_seconds),
            from_edge=tpl.from_edge,
        )
        for tpl in templates
    ]
    return rescaled, source_max_depart


def _sample_with_replacement(pool: Sequence[DemandTemplate], n: int, rng: random.Random) -> List[DemandTemplate]:
    if n <= 0 or not pool:
        return []
    if n <= len(pool):
        return rng.sample(list(pool), n)
    return [rng.choice(pool) for _ in range(n)]


def _expand_with_multiplier(
    templates: Sequence[DemandTemplate],
    multiplier_fn: Callable[[float], float],
    rng: random.Random,
    duration_seconds: float,
    jitter_seconds: float = 0.75,
) -> List[Tuple[DemandTemplate, float]]:
    expanded: List[Tuple[DemandTemplate, float]] = []

    for tpl in templates:
        multiplier = max(0.0, float(multiplier_fn(tpl.depart)))
        reps = int(math.floor(multiplier))
        fractional = multiplier - reps
        if rng.random() < fractional:
            reps += 1

        for rep_idx in range(reps):
            depart = tpl.depart + rng.uniform(-jitter_seconds, jitter_seconds) + (rep_idx * 0.01)
            depart = _clamp_depart(depart, duration_seconds)
            expanded.append((tpl, depart))

    return expanded


def _build_low_balanced(
    templates: Sequence[DemandTemplate],
    rng: random.Random,
    duration_seconds: float,
    scale: float = 0.30,
) -> List[Tuple[DemandTemplate, float]]:
    groups: Dict[str, List[DemandTemplate]] = defaultdict(list)
    for tpl in templates:
        groups[tpl.from_edge or "__unknown__"].append(tpl)

    target_total = max(1, int(round(len(templates) * scale)))
    group_keys = sorted(groups.keys())
    if not group_keys:
        return []

    per_group = target_total // len(group_keys)
    remainder = target_total % len(group_keys)

    selected: List[Tuple[DemandTemplate, float]] = []
    for idx, gk in enumerate(group_keys):
        take = per_group + (1 if idx < remainder else 0)
        for tpl in _sample_with_replacement(groups[gk], take, rng):
            selected.append((tpl, rng.uniform(0.0, duration_seconds - 0.01)))

    return selected


def _parse_net_geometry(net_file: Path) -> Dict[str, Dict[str, Any]]:
    tree = ET.parse(net_file)
    root = tree.getroot()

    junction_xy: Dict[str, Tuple[float, float]] = {}
    for junction in root.findall("junction"):
        j_id = junction.get("id")
        if not j_id:
            continue
        junction_xy[j_id] = (
            _safe_float(junction.get("x"), 0.0),
            _safe_float(junction.get("y"), 0.0),
        )

    edge_meta: Dict[str, Dict[str, Any]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue

        from_j = edge.get("from")
        to_j = edge.get("to")
        dx = 0.0
        dy = 0.0
        if from_j in junction_xy and to_j in junction_xy:
            x1, y1 = junction_xy[from_j]
            x2, y2 = junction_xy[to_j]
            dx = x2 - x1
            dy = y2 - y1

        lane_ids: List[str] = []
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            if lane_id:
                lane_ids.append(lane_id)

        lane_ids.sort(key=lambda lid: int(lid.rsplit("_", 1)[-1]) if lid.rsplit("_", 1)[-1].isdigit() else 9999)
        edge_meta[edge_id] = {"dx": dx, "dy": dy, "lane_ids": lane_ids}

    return edge_meta


def _select_major_eastbound_edge(
    templates: Sequence[DemandTemplate],
    net_file: Path,
) -> Tuple[str, str]:
    from_counts = Counter(t.from_edge for t in templates if t.from_edge)
    if not from_counts:
        raise RuntimeError("Cannot detect major flow edge: no 'from' edge in demand templates")

    edge_meta = _parse_net_geometry(net_file)

    eastbound_candidates = []
    for edge_id, count in from_counts.items():
        meta = edge_meta.get(edge_id)
        if not meta:
            continue
        dx = float(meta.get("dx", 0.0))
        dy = float(meta.get("dy", 0.0))
        if dx > 0.0 and abs(dx) >= abs(dy):
            eastbound_candidates.append((count, edge_id))

    if eastbound_candidates:
        eastbound_candidates.sort(reverse=True)
        major_edge = eastbound_candidates[0][1]
    else:
        major_edge = from_counts.most_common(1)[0][0]

    lane_ids = edge_meta.get(major_edge, {}).get("lane_ids", [])
    lane_id = lane_ids[0] if lane_ids else f"{major_edge}_0"

    return major_edge, lane_id


def _build_skewed_85_15(
    templates: Sequence[DemandTemplate],
    major_edge: str,
    rng: random.Random,
    duration_seconds: float,
    total_scale: float = 1.40,
) -> List[Tuple[DemandTemplate, float]]:
    total_target = max(1, int(round(len(templates) * total_scale)))

    major_pool = [t for t in templates if t.from_edge == major_edge]
    minor_pool = [t for t in templates if t.from_edge != major_edge]
    if not major_pool or not minor_pool:
        return _expand_with_multiplier(
            templates,
            lambda _t: total_scale,
            rng,
            duration_seconds=duration_seconds,
        )

    major_target = int(round(total_target * 0.85))
    minor_target = max(0, total_target - major_target)

    selected: List[Tuple[DemandTemplate, float]] = []
    for tpl in _sample_with_replacement(major_pool, major_target, rng):
        selected.append((tpl, rng.uniform(0.0, duration_seconds - 0.01)))
    for tpl in _sample_with_replacement(minor_pool, minor_target, rng):
        selected.append((tpl, rng.uniform(0.0, duration_seconds - 0.01)))

    return selected


def _write_route_file(
    output_file: Path,
    root_attrib: Dict[str, str],
    non_demand_elements: Sequence[ET.Element],
    demand_records: Sequence[Tuple[DemandTemplate, float]],
    scenario_prefix: str,
    duration_seconds: float,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("routes", attrib=root_attrib)

    for elem in non_demand_elements:
        root.append(copy.deepcopy(elem))

    sorted_records = sorted(demand_records, key=lambda x: x[1])
    for idx, (tpl, depart) in enumerate(sorted_records):
        elem = copy.deepcopy(tpl.element)
        base_id = elem.get("id", f"veh{idx}")
        elem.set("id", f"{scenario_prefix}_{idx}_{base_id}")
        elem.set("depart", f"{_clamp_depart(depart, duration_seconds):.2f}")
        root.append(elem)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    return len(sorted_records)


def _write_lane_blockage_additional(
    output_file: Path,
    edge_id: str,
    lane_id: str,
    begin: int,
    end: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "additional",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )

    rerouter = ET.SubElement(
        root,
        "rerouter",
        attrib={
            "id": "suite_lane_blockage",
            "edges": edge_id,
            "prob": "1.0",
        },
    )

    interval = ET.SubElement(
        rerouter,
        "interval",
        attrib={"begin": str(begin), "end": str(end)},
    )

    ET.SubElement(interval, "closingLaneReroute", attrib={"id": lane_id})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _check(
    name: str,
    lhs: Optional[float],
    rhs: Optional[float],
    factor: float,
    op: str,
) -> Dict[str, Any]:
    if lhs is None or rhs is None:
        return {
            "name": name,
            "passed": None,
            "lhs": lhs,
            "rhs": rhs,
            "factor": factor,
            "operator": op,
            "threshold": None,
            "reason": "missing metric",
        }

    threshold = rhs * factor
    if op == ">=":
        passed = lhs >= threshold
    elif op == "<=":
        passed = lhs <= threshold
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return {
        "name": name,
        "passed": passed,
        "lhs": lhs,
        "rhs": rhs,
        "factor": factor,
        "operator": op,
        "threshold": threshold,
    }


def _assess_scenario(
    scenario_id: str,
    rl_result: Optional[Dict[str, Any]],
    fixed_result: Optional[Dict[str, Any]],
    mp_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fixed_result = fixed_result or {}
    mp_result = mp_result or {}

    if rl_result is None:
        return {
            "status": "skipped",
            "checks": [],
            "note": "RL checkpoint not provided",
        }

    b_wait = [
        _num(fixed_result.get("mean_waiting_time")),
        _num(mp_result.get("mean_waiting_time")),
    ]
    b_wait = [v for v in b_wait if v is not None]
    best_wait = min(b_wait) if b_wait else None

    b_thr = [
        _num(fixed_result.get("mean_throughput")),
        _num(mp_result.get("mean_throughput")),
    ]
    b_thr = [v for v in b_thr if v is not None]
    best_thr = max(b_thr) if b_thr else None

    b_speed = [
        _num(fixed_result.get("mean_avg_speed")),
        _num(mp_result.get("mean_avg_speed")),
    ]
    b_speed = [v for v in b_speed if v is not None]
    best_speed = max(b_speed) if b_speed else None

    b_halts = [
        _num(fixed_result.get("mean_total_halts")),
        _num(mp_result.get("mean_total_halts")),
    ]
    b_halts = [v for v in b_halts if v is not None]
    best_halts = min(b_halts) if b_halts else None

    rl_wait = _num(rl_result.get("mean_waiting_time"))
    rl_thr = _num(rl_result.get("mean_throughput"))
    rl_speed = _num(rl_result.get("mean_avg_speed"))
    rl_halts = _num(rl_result.get("mean_total_halts"))

    checks: List[Dict[str, Any]] = []

    if scenario_id == "test1_stability_low_balanced":
        checks.append(_check("waiting <= 1.05 * best_baseline", rl_wait, best_wait, 1.05, "<="))
        checks.append(_check("halts <= 1.10 * best_baseline", rl_halts, best_halts, 1.10, "<="))
    elif scenario_id == "test2_saturation_high":
        checks.append(_check("throughput >= 0.98 * best_baseline", rl_thr, best_thr, 0.98, ">="))
        checks.append(_check("halts <= 1.15 * best_baseline", rl_halts, best_halts, 1.15, "<="))
    elif scenario_id == "test3_spatial_skew_85_15":
        checks.append(_check("throughput >= 0.95 * best_baseline", rl_thr, best_thr, 0.95, ">="))
        checks.append(_check("waiting <= 1.15 * best_baseline", rl_wait, best_wait, 1.15, "<="))
    elif scenario_id == "test4_temporal_shock":
        checks.append(_check("throughput >= 0.95 * best_baseline", rl_thr, best_thr, 0.95, ">="))
        checks.append(_check("speed >= 0.95 * best_baseline", rl_speed, best_speed, 0.95, ">="))
    elif scenario_id == "test5_cyclic_sinusoidal":
        checks.append(_check("speed >= 0.95 * best_baseline", rl_speed, best_speed, 0.95, ">="))
        checks.append(_check("waiting <= 1.10 * best_baseline", rl_wait, best_wait, 1.10, "<="))
    elif scenario_id == "test6_lane_blockage":
        checks.append(_check("throughput >= 0.90 * best_baseline", rl_thr, best_thr, 0.90, ">="))
        checks.append(_check("waiting <= 1.20 * best_baseline", rl_wait, best_wait, 1.20, "<="))
    else:
        checks.append({"name": "unsupported scenario", "passed": None})

    hard_fails = [c for c in checks if c.get("passed") is False]
    unknown = [c for c in checks if c.get("passed") is None]

    if hard_fails:
        status = "fail"
    elif unknown:
        status = "inconclusive"
    else:
        status = "pass"

    return {
        "status": status,
        "checks": checks,
        "note": (
            "Criteria use available aggregate metrics. "
            "Switch-rate/starvation/spillback directional checks need additional per-direction instrumentation."
        ),
    }


def _generate_scenarios(
    trip_source_file: Path,
    output_dir: Path,
    net_file: Path,
    duration_seconds: int,
    seed: int,
    extra_non_demand_elements: Optional[Sequence[ET.Element]] = None,
    selected_scenarios: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)

    root_attrib, non_demand, templates = _load_route_templates(trip_source_file)
    if not templates:
        raise RuntimeError(f"No trip/vehicle templates found in {trip_source_file}")

    non_demand_merged = _merge_non_demand_elements(
        non_demand,
        list(extra_non_demand_elements or []),
    )

    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")

    templates, source_max_depart = _rescale_depart_horizon(
        templates,
        duration_seconds=float(duration_seconds),
    )

    major_edge, blocked_lane_id = _select_major_eastbound_edge(templates, net_file)

    scenario_specs = [
        ScenarioSpec(
            scenario_id="test1_stability_low_balanced",
            test_name="Test 1",
            description="Low volume (30%) with balanced origins",
        ),
        ScenarioSpec(
            scenario_id="test2_saturation_high",
            test_name="Test 2",
            description="High saturation demand",
        ),
        ScenarioSpec(
            scenario_id="test3_spatial_skew_85_15",
            test_name="Test 3",
            description="Spatial skew demand: 85/15 major/minor flow",
        ),
        ScenarioSpec(
            scenario_id="test4_temporal_shock",
            test_name="Test 4",
            description="Temporal shock at 1/3 horizon",
        ),
        ScenarioSpec(
            scenario_id="test5_cyclic_sinusoidal",
            test_name="Test 5",
            description="Sinusoidal low-high-low demand",
        ),
        ScenarioSpec(
            scenario_id="test6_lane_blockage",
            test_name="Test 6",
            description="Temporary lane blockage (600s) at 1/3 horizon",
        ),
    ]

    selected_set = set(selected_scenarios) if selected_scenarios else None
    generated: Dict[str, Dict[str, Any]] = {}

    for spec in scenario_specs:
        if selected_set is not None and spec.scenario_id not in selected_set:
            continue

        if spec.scenario_id == "test1_stability_low_balanced":
            demand_records = _build_low_balanced(
                templates,
                rng=rng,
                duration_seconds=float(duration_seconds),
                scale=0.30,
            )
        elif spec.scenario_id == "test2_saturation_high":
            demand_records = _expand_with_multiplier(
                templates,
                lambda _t: 1.70,
                rng=rng,
                duration_seconds=float(duration_seconds),
            )
        elif spec.scenario_id == "test3_spatial_skew_85_15":
            demand_records = _build_skewed_85_15(
                templates,
                major_edge=major_edge,
                rng=rng,
                duration_seconds=float(duration_seconds),
                total_scale=1.40,
            )
        elif spec.scenario_id == "test4_temporal_shock":
            shock_t = duration_seconds / 3.0
            demand_records = _expand_with_multiplier(
                templates,
                lambda t: 0.75 if t < shock_t else 1.90,
                rng=rng,
                duration_seconds=float(duration_seconds),
            )
        elif spec.scenario_id == "test5_cyclic_sinusoidal":
            demand_records = _expand_with_multiplier(
                templates,
                lambda t: max(
                    0.20,
                    1.00 + 0.80 * math.sin((2.0 * math.pi * t / duration_seconds) - (math.pi / 2.0)),
                ),
                rng=rng,
                duration_seconds=float(duration_seconds),
            )
        elif spec.scenario_id == "test6_lane_blockage":
            demand_records = _expand_with_multiplier(
                templates,
                lambda _t: 1.00,
                rng=rng,
                duration_seconds=float(duration_seconds),
            )
        else:
            demand_records = []

        scenario_route = output_dir / f"{spec.scenario_id}.rou.xml"
        scenario_count = _write_route_file(
            output_file=scenario_route,
            root_attrib=root_attrib,
            non_demand_elements=non_demand_merged,
            demand_records=demand_records,
            scenario_prefix=spec.scenario_id,
            duration_seconds=float(duration_seconds),
        )

        extra_additional_file = None
        if spec.scenario_id == "test6_lane_blockage":
            begin = int(duration_seconds / 3)
            end = min(int(duration_seconds), int(begin + 600))
            if end <= begin:
                end = begin + 1
            extra_additional_file = output_dir / f"{spec.scenario_id}.add.xml"
            _write_lane_blockage_additional(
                output_file=extra_additional_file,
                edge_id=major_edge,
                lane_id=blocked_lane_id,
                begin=begin,
                end=end,
            )

        generated[spec.scenario_id] = {
            "test_name": spec.test_name,
            "description": spec.description,
            "route_file": str(scenario_route.resolve()),
            "extra_additional_file": str(extra_additional_file.resolve()) if extra_additional_file else None,
            "generated_trip_count": scenario_count,
        }

    generation_meta = {
        "trip_source_file": str(trip_source_file.resolve()),
        "trip_source_max_depart": source_max_depart,
        "scenario_duration_seconds": duration_seconds,
        "major_eastbound_edge": major_edge,
        "blocked_lane_id": blocked_lane_id,
        "seed": seed,
    }

    return generated, generation_meta


def run_suite(
    checkpoint: Optional[str],
    network: str,
    seeds: List[int],
    selected_scenarios: Optional[List[str]],
    baseline_mode: str,
    config_path: Optional[str],
    mp_net_info: Optional[str],
    output_file: Path,
    scenario_output_dir: Path,
    generate_only: bool,
) -> Dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    yaml_cfg = load_model_config(config_path)
    network_cfg = get_network_config(yaml_cfg, PROJECT_ROOT)
    env_cfg = get_env_config(yaml_cfg)
    duration_seconds = int(env_cfg.get("num_seconds", 3600))

    # Respect CLI network override
    if network != network_cfg["network_name"]:
        override = {"network": {"name": network}}
        network_cfg = get_network_config(override, PROJECT_ROOT)

    route_files = [Path(p.strip()) for p in str(network_cfg["route_file"]).split(",") if p.strip()]
    if not route_files:
        raise RuntimeError("No route files found in network configuration")

    trip_source = _detect_trip_source(route_files)
    type_route_files = [rf for rf in route_files if rf != trip_source]
    net_file = Path(network_cfg["net_file"])

    extra_non_demand: List[ET.Element] = []
    for rf in type_route_files:
        if rf.exists():
            extra_non_demand.extend(_load_non_demand_elements(rf))

    generated, generation_meta = _generate_scenarios(
        trip_source_file=trip_source,
        output_dir=scenario_output_dir,
        net_file=net_file,
        duration_seconds=duration_seconds,
        seed=seeds[0] if seeds else 42,
        extra_non_demand_elements=extra_non_demand,
        selected_scenarios=selected_scenarios,
    )

    report: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network": network,
        "checkpoint": checkpoint,
        "seeds": seeds,
        "selected_scenarios": selected_scenarios,
        "baseline_mode": baseline_mode,
        "generation": {
            "meta": generation_meta,
            "scenarios": generated,
        },
        "results": {},
    }

    if generate_only:
        report["summary"] = {"status": "generated_only"}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    scenarios_to_run = set(selected_scenarios) if selected_scenarios else set(generated.keys())

    for scenario_id, scenario_info in generated.items():
        if scenario_id not in scenarios_to_run:
            continue

        scenario_route = scenario_info["route_file"]
        scenario_additional = scenario_info["extra_additional_file"]

        route_override_parts = [str(p.resolve()) for p in type_route_files] + [scenario_route]
        route_override = ",".join(route_override_parts)

        fixed_output = output_file.parent / f"{scenario_id}_fixed.json"
        mp_output = output_file.parent / f"{scenario_id}_max_pressure.json"
        rl_output = output_file.parent / f"{scenario_id}_rl.json"

        print("\n" + "=" * 100)
        print(f"Running scenario: {scenario_id}")
        print("=" * 100)

        fixed_result: Optional[Dict[str, Any]] = None
        mp_result: Optional[Dict[str, Any]] = None

        if baseline_mode in {"both", "fixed"}:
            fixed_result = evaluate_baseline(
                network_name=network,
                num_episodes=len(seeds),
                use_gui=False,
                render=False,
                output_file=str(fixed_output),
                seeds=seeds,
                config_path=config_path,
                controller="fixed",
                mp_net_info=mp_net_info,
                route_files_override=route_override,
                extra_additional_files=scenario_additional,
            )

        if baseline_mode in {"both", "max_pressure_native"}:
            mp_result = evaluate_baseline(
                network_name=network,
                num_episodes=len(seeds),
                use_gui=False,
                render=False,
                output_file=str(mp_output),
                seeds=seeds,
                config_path=config_path,
                controller="max_pressure_native",
                mp_net_info=mp_net_info,
                route_files_override=route_override,
                extra_additional_files=scenario_additional,
            )

        rl_result = None
        if checkpoint:
            rl_result = evaluate_mgmq(
                checkpoint_path=checkpoint,
                network_name=network,
                num_episodes=len(seeds),
                use_gui=False,
                render=False,
                output_file=str(rl_output),
                seeds=seeds,
                use_training_config=True,
                config_path=config_path,
                cycle_time_override=None,
                route_files_override=route_override,
                extra_additional_files=scenario_additional,
            )

        assessment = _assess_scenario(
            scenario_id=scenario_id,
            rl_result=rl_result,
            fixed_result=fixed_result,
            mp_result=mp_result,
        )

        report["results"][scenario_id] = {
            "scenario": scenario_info,
            "fixed": fixed_result,
            "max_pressure_native": mp_result,
            "rl": rl_result,
            "assessment": assessment,
        }

    statuses = [entry["assessment"]["status"] for entry in report["results"].values()]
    report["summary"] = {
        "pass": statuses.count("pass"),
        "fail": statuses.count("fail"),
        "inconclusive": statuses.count("inconclusive"),
        "skipped": statuses.count("skipped"),
        "total": len(statuses),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized 6-scenario robustness suite")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional RL checkpoint for MGMQ evaluation")
    parser.add_argument("--network", type=str, default="zurich", help="Network name")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Evaluation seeds (one episode per seed)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to model_config.yml")
    parser.add_argument("--mp-net-info", type=str, default=None, help="Optional explicit MP net-info path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--scenario-output-dir",
        type=str,
        default=None,
        help="Directory to write generated scenario route/additional files",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate scenario files and manifest, skip evaluations",
    )
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--fixed-only",
        action="store_true",
        help="Run only fixed baseline controller",
    )
    baseline_group.add_argument(
        "--mp-only",
        action="store_true",
        help="Run only max_pressure_native baseline controller",
    )
    parser.add_argument("--test1", action="store_true", help="Run only Test 1 (stability low balanced)")
    parser.add_argument("--test2", action="store_true", help="Run only Test 2 (saturation high)")
    parser.add_argument("--test3", action="store_true", help="Run only Test 3 (spatial skew 85/15)")
    parser.add_argument("--test4", action="store_true", help="Run only Test 4 (temporal shock)")
    parser.add_argument("--test5", action="store_true", help="Run only Test 5 (cyclic sinusoidal)")
    parser.add_argument("--test6", action="store_true", help="Run only Test 6 (lane blockage)")

    args = parser.parse_args()

    selected_scenarios: List[str] = []
    if args.test1:
        selected_scenarios.append("test1_stability_low_balanced")
    if args.test2:
        selected_scenarios.append("test2_saturation_high")
    if args.test3:
        selected_scenarios.append("test3_spatial_skew_85_15")
    if args.test4:
        selected_scenarios.append("test4_temporal_shock")
    if args.test5:
        selected_scenarios.append("test5_cyclic_sinusoidal")
    if args.test6:
        selected_scenarios.append("test6_lane_blockage")

    baseline_mode = "both"
    if args.fixed_only:
        baseline_mode = "fixed"
    elif args.mp_only:
        baseline_mode = "max_pressure_native"

    output_file = Path(args.output) if args.output else (
        PROJECT_ROOT / "results_mgmq_test" / f"six_scenario_suite_{args.network}_latest.json"
    )
    scenario_output_dir = Path(args.scenario_output_dir) if args.scenario_output_dir else (
        PROJECT_ROOT / "network" / args.network / "six_scenario_suite"
    )

    report = run_suite(
        checkpoint=args.checkpoint,
        network=args.network,
        seeds=list(args.seeds),
        selected_scenarios=selected_scenarios or None,
        baseline_mode=baseline_mode,
        config_path=args.config,
        mp_net_info=args.mp_net_info,
        output_file=output_file,
        scenario_output_dir=scenario_output_dir,
        generate_only=args.generate_only,
    )

    print("\n" + "=" * 100)
    print("SIX-SCENARIO SUITE COMPLETED")
    print("=" * 100)
    print(f"Report: {output_file}")
    print(json.dumps(report.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
