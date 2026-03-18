#!/usr/bin/env python3
"""
Enhanced script to extract traffic light information from SUMO .net.xml file
and create a net-info.json file similar to the phuquoc format
"""

import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path
import re

def parse_connections_from_tllogic(tl_logic, tl_junctions):
    """Parse connections and movements from traffic light logic"""
    tl_id = tl_logic.get('id')
    if tl_id not in tl_junctions:
        return {}, {}
    
    inc_lanes = tl_junctions[tl_id]['incLanes']
    
    # Extract edge IDs from lane IDs
    edges = set()
    lane_to_edge = {}
    for lane_id in inc_lanes:
        # Handle different lane ID formats
        if '_' in lane_id:
            edge_id = lane_id.rsplit('_', 1)[0]
        else:
            edge_id = lane_id
        edges.add(edge_id)
        lane_to_edge[lane_id] = edge_id
    
    movements = {}
    phases_info = {}
    
    # Analyze phases
    for i, phase in enumerate(tl_logic.findall('phase')):
        state = phase.get('state', '')
        duration = int(phase.get('duration', 30))
        
        # Analyze which movements are allowed in this phase
        phase_movements = []
        for j, signal in enumerate(state):
            if signal in ['G', 'g'] and j < len(inc_lanes):
                # This lane has green signal
                from_lane = inc_lanes[j]
                from_edge = lane_to_edge.get(from_lane, from_lane)
                
                # For now, create a simple movement structure
                # In a real implementation, you'd need connection information
                phase_movements.append([from_edge, "unknown_target"])
        
        phases_info[str(i)] = {
            'movements': phase_movements,
            'duration': duration,
            'min-green': 15,
            'max-green': 120
        }
    
    # Create basic movements (equal probability)
    for edge in edges:
        if edge not in movements:
            movements[edge] = {}
        
        # Create movements to other edges (simplified)
        other_edges = [e for e in edges if e != edge]
        if other_edges:
            prob = 1.0 / len(other_edges) if other_edges else 0
            for target_edge in other_edges:
                movements[edge][target_edge] = round(prob, 6)
    
    return movements, phases_info

def extract_net_info_enhanced(net_file_path):
    """Extract traffic light information from SUMO network file with enhanced analysis"""
    
    # Parse the XML file
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    
    net_info = {"tls": {}}
    
    # Find all traffic light junctions
    tl_junctions = {}
    for junction in root.findall('junction'):
        if junction.get('type') == 'traffic_light':
            tl_id = junction.get('id')
            inc_lanes_str = junction.get('incLanes', '')
            inc_lanes = inc_lanes_str.split() if inc_lanes_str else []
            
            tl_junctions[tl_id] = {
                'incLanes': inc_lanes,
                'intLanes': junction.get('intLanes', '').split(),
                'x': float(junction.get('x', 0)),
                'y': float(junction.get('y', 0))
            }
    
    print(f"Found {len(tl_junctions)} traffic light junctions")
    
    # Find all edges with their properties
    edges = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):  # Skip internal edges
            lanes = edge.findall('lane')
            if lanes:
                # Get properties from the first lane (or aggregate if needed)
                first_lane = lanes[0]
                speed = float(first_lane.get('speed', 13.89))
                length = float(first_lane.get('length', 0))
                
                edges[edge_id] = {
                    'speed': speed,
                    'length': length,
                    'lanes': len(lanes)
                }
    
    print(f"Found {len(edges)} edges")
    
    # Find traffic light programs and build detailed info
    for tl_logic in root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        
        if tl_id not in tl_junctions:
            continue
        
        # Calculate cycle time
        phases = tl_logic.findall('phase')
        cycle_time = sum(int(phase.get('duration', 30)) for phase in phases)
        
        # Get edges connected to this traffic light
        inc_lanes = tl_junctions[tl_id]['incLanes']
        connected_edges = set()
        
        for lane_id in inc_lanes:
            # Extract edge ID from lane ID
            if '_' in lane_id:
                edge_id = lane_id.rsplit('_', 1)[0]
            else:
                edge_id = lane_id
            
            if edge_id in edges:
                connected_edges.add(edge_id)
        
        print(f"TL {tl_id}: {len(connected_edges)} connected edges")
        
        # Build edge information for this traffic light
        tl_edges = {}
        for edge_id in connected_edges:
            edge = edges[edge_id]
            
            # Determine saturation flow based on number of lanes and speed
            lanes_count = edge['lanes']
            if edge['speed'] > 25:  # High speed roads
                sat_flow = lanes_count * 1800
            elif edge['speed'] > 15:  # Medium speed roads  
                sat_flow = lanes_count * 1500
            else:  # Low speed roads
                sat_flow = lanes_count * 1200
            
            tl_edges[edge_id] = {
                'sat_flow': sat_flow,
                'length': round(edge['length'], 2),
                'speed': round(edge['speed'], 2),
                'detector': []  # Empty for now, would need detector files
            }
        
        # Parse movements and phases
        movements, phases_info = parse_connections_from_tllogic(tl_logic, tl_junctions)
        
        # Determine controller type based on TL ID
        if tl_id and (tl_id in ['1', '2', '3', '4'] or tl_id.startswith('J')):
            controller = 'fixed_time'
        else:
            controller = 'max_pressure'  # For cluster intersections
        
        # Build the traffic light info
        tl_info = {
            'cycle': cycle_time,
            'controller': controller,
            'edges': tl_edges,
            'movements': movements,
            'phases': phases_info
        }
        
        net_info['tls'][tl_id] = tl_info
    
    return net_info

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_net_info_enhanced.py <input_net_file> <output_json_file>")
        sys.exit(1)
    
    net_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(net_file).exists():
        print(f"Error: Input file {net_file} does not exist")
        sys.exit(1)
    
    try:
        net_info = extract_net_info_enhanced(net_file)
        
        # Write the output JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(net_info, f, indent=2)
        
        print(f"\nSuccessfully created {output_file}")
        print(f"Found {len(net_info['tls'])} traffic light intersections")
        
        # Print summary
        for tl_id, tl_data in net_info['tls'].items():
            print(f"  TL {tl_id}: {len(tl_data['edges'])} edges, {len(tl_data['phases'])} phases, "
                  f"cycle={tl_data['cycle']}s, controller={tl_data['controller']}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()