"""Generate low-demand and high-demand route file variants from an existing SUMO route file.

Low demand  (--scale 0.5): keep 50% of vehicles (random sample, seed-stable).
High demand (--scale 1.5): keep all vehicles + duplicate 50% with small depart offset.

Usage:
    # Generate both low and high for arterial4x4:
    python scripts/generate_demand_variants.py \\
        --input network/arterial4x4/arterial4x4_1.rou.xml \\
        --output-low  network/arterial4x4/arterial4x4_low.rou.xml \\
        --output-high network/arterial4x4/arterial4x4_high.rou.xml

    # Generate both low and high for resco_grid4x4:
    python scripts/generate_demand_variants.py \\
        --input network/resco_grid4x4/resco_grid4x4.rou.xml \\
        --output-low  network/resco_grid4x4/resco_grid4x4_low.rou.xml \\
        --output-high network/resco_grid4x4/resco_grid4x4_high.rou.xml
"""

import argparse
import random
import copy
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def _indent(elem, level=0):
    """Add pretty-print indentation in-place (ElementTree doesn't do this by default)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def load_route_file(path: str):
    """Parse SUMO route XML, return (root, vtype_elem, vehicle_list)."""
    tree = ET.parse(path)
    root = tree.getroot()
    vtypes = root.findall("vType")
    vehicles = root.findall("vehicle")
    trips = root.findall("trip")
    return root, vtypes, vehicles + trips


def write_route_file(output_path: str, vtypes, vehicles, original_root):
    """Write route file preserving original schema attributes."""
    root_attrib = original_root.attrib.copy()
    new_root = ET.Element("routes", attrib=root_attrib)

    # Preserve vType definitions
    for vt in vtypes:
        new_root.append(copy.deepcopy(vt))

    # Add vehicles (already sorted by depart time)
    for v in sorted(vehicles, key=lambda x: float(x.get("depart", 0))):
        new_root.append(copy.deepcopy(v))

    _indent(new_root)
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="  ")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output), encoding="utf-8", xml_declaration=True)
    return len(vehicles)


def generate_low_demand(vehicles, scale: float = 0.5, seed: int = 42):
    """Return a random sample of `scale` fraction of vehicles (stable seed)."""
    rng = random.Random(seed)
    n_keep = max(1, round(len(vehicles) * scale))
    sampled = rng.sample(vehicles, n_keep)
    return sampled


def generate_high_demand(vehicles, scale: float = 1.5, seed: int = 42, depart_offset: float = 0.5):
    """Return all original vehicles + duplicates to reach `scale` total.

    Duplicate vehicles get IDs prefixed with 'xh_' and departure times
    shifted by `depart_offset` seconds to avoid conflicts.
    """
    rng = random.Random(seed)
    n_extra = round(len(vehicles) * (scale - 1.0))
    extras_src = rng.sample(vehicles, min(n_extra, len(vehicles)))

    # If scale > 2.0, repeat sampling
    extras = list(extras_src)
    while len(extras) < n_extra:
        more = rng.sample(vehicles, min(n_extra - len(extras), len(vehicles)))
        extras.extend(more)
    extras = extras[:n_extra]

    new_vehicles = list(vehicles)
    for i, v in enumerate(extras):
        nv = copy.deepcopy(v)
        nv.set("id", f"xh_{i}_{v.get('id', str(i))}")
        orig_depart = float(v.get("depart", 0))
        # Offset depart so vehicles don't appear at exact same time
        nv.set("depart", f"{orig_depart + depart_offset:.2f}")
        new_vehicles.append(nv)

    return new_vehicles


def main():
    parser = argparse.ArgumentParser(description="Generate low/high demand SUMO route file variants")
    parser.add_argument("--input", required=True, help="Input route file (.rou.xml)")
    parser.add_argument("--output-low", default=None,
                        help="Output path for low-demand variant (default: auto)")
    parser.add_argument("--output-high", default=None,
                        help="Output path for high-demand variant (default: auto)")
    parser.add_argument("--scale-low", type=float, default=0.5,
                        help="Fraction of vehicles for low demand (default: 0.5 = 50%%)")
    parser.add_argument("--scale-high", type=float, default=1.5,
                        help="Scale factor for high demand (default: 1.5 = 150%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--depart-offset", type=float, default=0.5,
                        help="Departure time offset (s) for duplicated vehicles in high demand")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    stem = input_path.stem  # e.g. "arterial4x4_1"
    parent = input_path.parent

    out_low = Path(args.output_low) if args.output_low else parent / f"{stem}_low_demand.rou.xml"
    out_high = Path(args.output_high) if args.output_high else parent / f"{stem}_high_demand.rou.xml"

    print(f"Input:  {input_path}")
    root, vtypes, vehicles = load_route_file(str(input_path))
    print(f"  Total vehicles: {len(vehicles)}")
    departs = [float(v.get("depart", 0)) for v in vehicles]
    print(f"  Depart range: [{min(departs):.1f}s, {max(departs):.1f}s]")

    # Generate low demand
    low_vehicles = generate_low_demand(vehicles, scale=args.scale_low, seed=args.seed)
    n_low = write_route_file(str(out_low), vtypes, low_vehicles, root)
    print(f"\nLow demand ({args.scale_low*100:.0f}%):")
    print(f"  Vehicles: {n_low} (from {len(vehicles)})")
    print(f"  Output:   {out_low}")

    # Generate high demand
    high_vehicles = generate_high_demand(
        vehicles, scale=args.scale_high, seed=args.seed, depart_offset=args.depart_offset
    )
    n_high = write_route_file(str(out_high), vtypes, high_vehicles, root)
    print(f"\nHigh demand ({args.scale_high*100:.0f}%):")
    print(f"  Vehicles: {n_high} (from {len(vehicles)})")
    print(f"  Output:   {out_high}")


if __name__ == "__main__":
    main()
