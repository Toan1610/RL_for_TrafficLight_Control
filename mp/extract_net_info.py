#!/usr/bin/env python3
"""
Script to extract traffic light information from SUMO .net.xml file
and create a net-info.json file similar to the phuquoc format
"""

import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path

def extract_net_info(net_file_path):
    """Extract traffic light information from SUMO network file"""
    
    # Parse the XML file
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    
    net_info = {"tls": {}}
    
    # Find all traffic light junctions
    tl_junctions = {}
    for junction in root.findall('junction'):
        if junction.get('type') == 'traffic_light':
            tl_id = junction.get('id')
            tl_junctions[tl_id] = {
                'incLanes': junction.get('incLanes', '').split(),
                'intLanes': junction.get('intLanes', '').split(),
                'x': float(junction.get('x', 0)),
                'y': float(junction.get('y', 0))
            }
    
    print(f"Found {len(tl_junctions)} traffic light junctions: {list(tl_junctions.keys())}")
    
    # Find all edges
    edges = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):  # Skip internal edges
            edge_info = {
                'id': edge_id,
                'lanes': []
            }
            
            # Get lane information
            for lane in edge.findall('lane'):
                lane_info = {
                    'id': lane.get('id'),
                    'speed': float(lane.get('speed', 13.89)),
                    'length': float(lane.get('length', 0))
                }
                edge_info['lanes'].append(lane_info)
            
            edges[edge_id] = edge_info
    
    print(f"Found {len(edges)} edges")
    
    # Find traffic light programs
    tl_programs = {}
    for tl_logic in root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        program_id = tl_logic.get('programID', '0')
        
        phases = []
        for phase in tl_logic.findall('phase'):
            phase_info = {
                'duration': int(phase.get('duration', 30)),
                'state': phase.get('state', ''),
                'minDur': int(phase.get('minDur', 15)),
                'maxDur': int(phase.get('maxDur', 120))
            }
            phases.append(phase_info)
        
        tl_programs[tl_id] = {
            'type': tl_logic.get('type', 'static'),
            'programID': program_id,
            'offset': int(tl_logic.get('offset', 0)),
            'phases': phases
        }
    
    print(f"Found {len(tl_programs)} traffic light programs: {list(tl_programs.keys())}")
    
    # Find connections for movement information
    connections = {}
    for connection in root.findall('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        from_lane = connection.get('fromLane')
        to_lane = connection.get('toLane')
        
        if from_edge and to_edge and not from_edge.startswith(':'):
            if from_edge not in connections:
                connections[from_edge] = {}
            if to_edge not in connections[from_edge]:
                connections[from_edge][to_edge] = []
            
            connections[from_edge][to_edge].append({
                'fromLane': from_lane,
                'toLane': to_lane
            })
    
    print(f"Found {len(connections)} edge connections")
    
    # Build the net-info structure for each traffic light
    for tl_id in tl_junctions:
        if tl_id in tl_programs:
            tl_info = {
                'cycle': sum(phase['duration'] for phase in tl_programs[tl_id]['phases']),
                'controller': 'fixed_time',  # Default controller type
                'edges': {},
                'movements': {},
                'phases': {}
            }
            
            # Get edges connected to this traffic light
            inc_lanes = tl_junctions[tl_id]['incLanes']
            connected_edges = set()
            
            for lane_id in inc_lanes:
                edge_id = lane_id.split('_')[0]  # Extract edge ID from lane ID
                if edge_id in edges:
                    connected_edges.add(edge_id)
            
            # Build edge information
            for edge_id in connected_edges:
                edge = edges[edge_id]
                tl_info['edges'][edge_id] = {
                    'sat_flow': 1800,  # Default saturation flow
                    'length': max(lane['length'] for lane in edge['lanes']) if edge['lanes'] else 0,
                    'speed': max(lane['speed'] for lane in edge['lanes']) if edge['lanes'] else 13.89,
                    'detector': []  # Will be populated if detector files exist
                }
            
            # Build movement information from connections
            for from_edge in connected_edges:
                if from_edge in connections:
                    tl_info['movements'][from_edge] = {}
                    total_movements = len(connections[from_edge])
                    if total_movements > 0:
                        prob_per_movement = 1.0 / total_movements
                        for to_edge in connections[from_edge]:
                            if to_edge in connected_edges or to_edge.startswith('-'):
                                tl_info['movements'][from_edge][to_edge] = round(prob_per_movement, 6)
            
            # Build phase information
            for i, phase in enumerate(tl_programs[tl_id]['phases']):
                tl_info['phases'][str(i)] = {
                    'movements': [],  # This would need more complex analysis of the phase state
                    'duration': phase['duration'],
                    'min-green': phase.get('minDur', 15),
                    'max-green': phase.get('maxDur', 120)
                }
            
            net_info['tls'][tl_id] = tl_info
    
    return net_info

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_net_info.py <input_net_file> <output_json_file>")
        sys.exit(1)
    
    net_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(net_file).exists():
        print(f"Error: Input file {net_file} does not exist")
        sys.exit(1)
    
    try:
        net_info = extract_net_info(net_file)
        
        # Write the output JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(net_info, f, indent=2)
        
        print(f"\nSuccessfully created {output_file}")
        print(f"Found {len(net_info['tls'])} traffic light intersections")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()