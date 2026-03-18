"""Generate high-traffic route file for grid4x4 network.

Creates ~60,000 vehicles/hour (3x original traffic) to create meaningful congestion
that requires intelligent signal control to manage.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    network_dir = Path(__file__).parent.parent / "network" / "grid4x4"
    net_file = network_dir / "grid4x4.net.xml"
    trip_file = network_dir / "grid4x4-heavy-trips.rou.xml"
    route_file = network_dir / "grid4x4-heavy.rou.xml"
    
    sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    duarouter = os.path.join(sumo_home, "bin", "duarouter")
    
    if not os.path.exists(random_trips):
        print(f"ERROR: randomTrips.py not found at {random_trips}")
        sys.exit(1)
    
    # Step 1: Generate trips with 3x traffic (60000 veh/h)
    print("Step 1: Generating high-traffic trips (60000 veh/h)...")
    cmd = [
        sys.executable, random_trips,
        "-n", str(net_file),
        "-o", str(trip_file),
        "--trip-attributes", 'type="td_0" departLane="best" departSpeed="max" departPos="random"',
        "--lanes",
        "--seed", "124",
        "--insertion-rate", "60000",
        "--random-depart",
        "--binomial", "10",
        "-b", "0",
        "-e", "3600",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"randomTrips STDOUT: {result.stdout}")
        print(f"randomTrips STDERR: {result.stderr}")
        # Try without --binomial (since period is too short for binomial)
        print("\nRetrying without --binomial...")
        cmd = [
            sys.executable, random_trips,
            "-n", str(net_file),
            "-o", str(trip_file),
            "--trip-attributes", 'type="td_0" departLane="best" departSpeed="max" departPos="random"',
            "--lanes",
            "--seed", "124",
            "--insertion-rate", "60000",
            "--random-depart",
            "-b", "0",
            "-e", "3600",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED: {result.stderr}")
            sys.exit(1)
    
    print(f"  Generated trips: {trip_file}")
    
    # Count trips
    with open(trip_file) as f:
        trip_count = sum(1 for line in f if '<trip ' in line)
    print(f"  Trip count: {trip_count}")
    
    # Step 2: Route trips through network using duarouter
    print("\nStep 2: Computing routes with duarouter...")
    cmd = [
        duarouter,
        "-n", str(net_file),
        "-t", str(trip_file),
        "-o", str(route_file),
        "--ignore-errors",
        "--no-warnings",
        "--no-step-log",
        "-b", "0",
        "-e", "3600",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"duarouter STDERR: {result.stderr}")
        # Still check if output was created
        if not route_file.exists():
            print("FAILED: No route file generated")
            sys.exit(1)
    
    # Count routes
    with open(route_file) as f:
        route_count = sum(1 for line in f if '<vehicle ' in line)
    print(f"  Generated routes: {route_file}")
    print(f"  Vehicle count: {route_count}")
    print(f"\nDone! Traffic intensity: ~{route_count / 3600:.1f} veh/sec")
    print(f"Original was ~{19799 / 3600:.1f} veh/sec (20000 insertion-rate)")
    print(f"New is ~{route_count / 3600:.1f} veh/sec (60000 insertion-rate)")
    
    # Also create the vType-only route file that references heavy routes
    # The sumocfg uses grid4x4.rou.xml (vTypes) + grid4x4-demo.rou.xml (trips)
    # We need to create a new heavy config that uses grid4x4.rou.xml + grid4x4-heavy.rou.xml
    print(f"\nTo use this traffic, update simulation.yml route_file to:")
    print(f"  route_file: network/grid4x4/grid4x4.rou.xml,network/grid4x4/grid4x4-heavy.rou.xml")


if __name__ == "__main__":
    main()
