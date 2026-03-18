#!/usr/bin/env python3
"""
Final enhanced script to extract traffic light information from SUMO .net.xml file
and create a net-info.json file matching the phuquoc format with detector information
"""

import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path


def edge_from_lane(lane_id):
    """Extract edge ID from SUMO lane ID."""
    if '_' in lane_id:
        return lane_id.rsplit('_', 1)[0]
    return lane_id

def parse_detectors(add_file_path):
    """Parse detector information from additional file"""
    detectors_by_edge = {}
    
    if not Path(add_file_path).exists():
        print(f"Additional file {add_file_path} not found, skipping detector parsing")
        return detectors_by_edge
    
    try:
        tree = ET.parse(add_file_path)
        root = tree.getroot()
        
        for detector in root.findall('laneAreaDetector'):
            detector_id = detector.get('id')
            lane_id = detector.get('lane')
            
            if lane_id and detector_id:
                # Extract edge ID from lane ID
                if '_' in lane_id:
                    edge_id = lane_id.rsplit('_', 1)[0]
                else:
                    edge_id = lane_id
                
                if edge_id not in detectors_by_edge:
                    detectors_by_edge[edge_id] = []
                
                detectors_by_edge[edge_id].append(detector_id)
        
        print(f"Found detectors for {len(detectors_by_edge)} edges")
        
    except Exception as e:
        print(f"Error parsing detectors: {e}")
    
    return detectors_by_edge

def extract_net_info_final(net_file_path, add_file_path=None):
    """Extract traffic light information with complete detector support"""
    
    # Parse detectors first
    detectors_by_edge = {}
    if add_file_path:
        detectors_by_edge = parse_detectors(add_file_path)
    
    # Parse the XML file
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    
    net_info = {"tls": {}}
    
    # Find all traffic light junctions, grouped by controller ID.
    # In SUMO, a junction ID can differ from the controlling tlLogic ID.
    tl_junctions = {}
    for junction in root.findall('junction'):
        if junction.get('type') == 'traffic_light':
            tl_id = junction.get('tl') or junction.get('id')
            if not tl_id:
                continue
            inc_lanes_str = junction.get('incLanes', '')
            inc_lanes = inc_lanes_str.split() if inc_lanes_str else []

            if tl_id not in tl_junctions:
                tl_junctions[tl_id] = {
                    'incLanes': [],
                    'intLanes': [],
                    'x': float(junction.get('x', 0)),
                    'y': float(junction.get('y', 0)),
                }

            tl_junctions[tl_id]['incLanes'].extend(inc_lanes)
            tl_junctions[tl_id]['intLanes'].extend(junction.get('intLanes', '').split())
    
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
    
    # Parse connections to build proper movements (from incoming to outgoing edges)
    connections_by_tl = {}
    for connection in root.findall('connection'):
        tl_id = connection.get('tl')
        if tl_id:  # Only consider connections controlled by traffic lights
            from_edge = connection.get('from')
            to_edge = connection.get('to')
            
            if from_edge and to_edge and not from_edge.startswith(':') and not to_edge.startswith(':'):
                if tl_id not in connections_by_tl:
                    connections_by_tl[tl_id] = {}
                
                if from_edge not in connections_by_tl[tl_id]:
                    connections_by_tl[tl_id][from_edge] = set()
                
                connections_by_tl[tl_id][from_edge].add(to_edge)
    
    print(f"Found connections for {len(connections_by_tl)} traffic lights")
    
    skipped_no_e2 = 0

    # Find traffic light programs and build detailed info
    for tl_logic in root.findall('tlLogic'):
        tl_id = tl_logic.get('id')

        if not tl_id:
            continue
        
        # Calculate cycle time
        phases = tl_logic.findall('phase')
        cycle_time = sum(int(phase.get('duration', 30)) for phase in phases)
        
        # Get edges connected to this traffic light
        inc_lanes = tl_junctions.get(tl_id, {}).get('incLanes', [])
        connected_edges = set()

        for lane_id in inc_lanes:
            edge_id = edge_from_lane(lane_id)
            if edge_id in edges:
                connected_edges.add(edge_id)

        # Fallback when incLanes are missing or junction IDs do not map directly
        if not connected_edges and tl_id in connections_by_tl:
            connected_edges.update(connections_by_tl[tl_id].keys())
        
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
            
            # Get detectors for this edge
            edge_detectors = detectors_by_edge.get(edge_id, [])
            
            tl_edges[edge_id] = {
                'sat_flow': sat_flow,
                'length': round(edge['length'], 2),
                'speed': round(edge['speed'], 2),
                'detector': edge_detectors
            }

        # Keep only controllable intersections with at least one E2 detector.
        # This matches the runtime environment where a TLS without E2 is skipped.
        detector_count = sum(len(edge_data['detector']) for edge_data in tl_edges.values())
        if add_file_path and detector_count == 0:
            skipped_no_e2 += 1
            print(f"TL {tl_id}: skipped (no E2 detectors)")
            continue
        
        # Build movements based on actual connections from incoming to outgoing edges
        movements = {}
        
        if tl_id in connections_by_tl:
            # Use actual connections from the network
            for from_edge, to_edges in connections_by_tl[tl_id].items():
                if from_edge in connected_edges:
                    movements[from_edge] = {}
                    to_edges_list = list(to_edges)
                    
                    if to_edges_list:
                        # Equal probability for each possible movement
                        prob = round(1.0 / len(to_edges_list), 6)
                        for to_edge in to_edges_list:
                            movements[from_edge][to_edge] = prob
        else:
            # Fallback: if no connections found, create movements between all edges
            # (this keeps backward compatibility but should rarely be used)
            for from_edge in connected_edges:
                movements[from_edge] = {}
                other_edges = [e for e in connected_edges if e != from_edge]
                if other_edges:
                    prob = round(1.0 / len(other_edges), 6)
                    for to_edge in other_edges:
                        movements[from_edge][to_edge] = prob
        
        # Build phase information with improved movement detection using connections
        # Only save phases with green lights (G or g), keep original phase index
        phases_info = {}
        
        # Build a mapping from lane to connection for this TL
        lane_to_connections = {}
        if tl_id in connections_by_tl:
            for connection in root.findall('connection'):
                if connection.get('tl') == tl_id:
                    from_edge = connection.get('from')
                    to_edge = connection.get('to')
                    link_index = connection.get('linkIndex')

                    if link_index and from_edge and to_edge and not from_edge.startswith(':') and not to_edge.startswith(':'):
                        link_idx = int(link_index)
                        if link_idx not in lane_to_connections:
                            lane_to_connections[link_idx] = []
                        lane_to_connections[link_idx].append((from_edge, to_edge))
        
        for i, phase in enumerate(phases):
            state = phase.get('state', '')
            duration = int(phase.get('duration', 30))
            
            # Check if this phase has any green lights (G or g)
            has_green = any(signal in ['G', 'g'] for signal in state)
            
            # Skip phases without green lights (transition phases)
            if not has_green:
                continue
            
            # Analyze which movements are allowed in this phase based on linkIndex
            phase_movements = []
            
            for j, signal in enumerate(state):
                if signal in ['G', 'g']:
                    # Check if this link index has connections
                    if j in lane_to_connections:
                        for from_edge, to_edge in lane_to_connections[j]:
                            if from_edge in connected_edges:
                                movement = [from_edge, to_edge]
                                if movement not in phase_movements:
                                    phase_movements.append(movement)
            
            # Keep original phase index from SUMO
            phases_info[str(i)] = {
                'movements': phase_movements,
                'duration': duration,
                'min-green': 5,
                'max-green': 120
            }

        # Compute feasible default min-green per green phase.
        lost_time = 0
        for phase in phases:
            state = phase.get('state', '')
            duration = int(phase.get('duration', 30))
            if not any(signal in ['G', 'g'] for signal in state):
                lost_time += duration

        green_phase_count = len(phases_info)
        available_green = max(1, cycle_time - lost_time)
        if green_phase_count > 0:
            base_min_green = max(1, min(15, available_green // green_phase_count))
            for phase_data in phases_info.values():
                phase_duration = max(1, int(phase_data['duration']))
                phase_data['min-green'] = min(base_min_green, phase_duration)
        
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

    if add_file_path:
        print(f"Active controlled traffic lights (with E2): {len(net_info['tls'])}")
        print(f"Skipped traffic lights without E2: {skipped_no_e2}")
    
    return net_info

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python extract_net_info_final.py <input_net_file> <output_json_file> [additional_file]")
        sys.exit(1)
    
    net_file = sys.argv[1]
    output_file = sys.argv[2]
    add_file = sys.argv[3] if len(sys.argv) == 4 else None
    
    if not Path(net_file).exists():
        print(f"Error: Input file {net_file} does not exist")
        sys.exit(1)
    
    try:
        net_info = extract_net_info_final(net_file, add_file)
        
        # Write the output JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(net_info, f, indent=2)
        
        print(f"\nSuccessfully created {output_file}")
        print(f"Found {len(net_info['tls'])} traffic light intersections")
        
        # Print summary
        for tl_id, tl_data in net_info['tls'].items():
            detector_count = sum(len(edge_data['detector']) for edge_data in tl_data['edges'].values())
            print(f"  TL {tl_id}: {len(tl_data['edges'])} edges, {len(tl_data['phases'])} phases, "
                  f"cycle={tl_data['cycle']}s, controller={tl_data['controller']}, {detector_count} detectors")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()