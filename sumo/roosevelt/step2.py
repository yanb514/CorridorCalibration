import xml.etree.ElementTree as ET
import pandas as pd

tmc2edges = {
    "107-13367": [], # W Roosevelt between Canal and Bridge
    "107N21071": [], # W Roosevelt Bridge
    "107P21071": [], # E Roosevelt Bridge
}
from math import radians, sin, cos, atan2, sqrt

def parse_sumo_net(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()

    location_elem = root.find('location')
    orig_boundary = list(map(float, location_elem.get('origBoundary').split(',')))
    conv_boundary = list(map(float, location_elem.get('convBoundary').split(',')))
    net_offset = list(map(float, location_elem.get('netOffset').split(',')))

    # Calculate scaling factors from origBoundary to convBoundary
    orig_width = orig_boundary[2] - orig_boundary[0]
    orig_height = orig_boundary[3] - orig_boundary[1]
    conv_width = conv_boundary[2] - conv_boundary[0]
    conv_height = conv_boundary[3] - conv_boundary[1]
    scale_x = conv_width / orig_width
    scale_y = conv_height / orig_height

    edges = []
    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        lanes = edge_elem.findall('lane')
        if not lanes:
            continue

        first_lane = lanes[0]
        shape = first_lane.get('shape')
        shape_points = [tuple(map(float, point.split(','))) for point in shape.split()]

        start_x, start_y = shape_points[0]
        end_x, end_y = shape_points[-1]
        length = float(first_lane.get('length'))

        # Convert SUMO local coordinates to geographical coordinates
        def sumo_to_latlon(x, y):
            # Reverse the netOffset
            x_utm = x + net_offset[0]
            y_utm = y + net_offset[1]
            # Scale back to original boundary
            lon = orig_boundary[0] + (x_utm - conv_boundary[0]) / scale_x
            lat = orig_boundary[1] + (y_utm - conv_boundary[1]) / scale_y
            return lat, lon

        start_lat, start_lon = sumo_to_latlon(start_x, start_y)
        end_lat, end_lon = sumo_to_latlon(end_x, end_y)

        edges.append({
            'id': edge_id,
            'start_lat': start_lat,
            'start_lon': start_lon,
            'end_lat': end_lat,
            'end_lon': end_lon,
            'length': length
        })

    return edges

def parse_tmc_data(tmc_csv):
    return pd.read_csv(tmc_csv)

def distance(lat1, lon1, lat2, lon2):
    # Approximate distance in meters using Haversine formula
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def is_direction_match(tmc_dir, edge_start_lat, edge_start_lon, edge_end_lat, edge_end_lon):
    dx = edge_end_lon - edge_start_lon
    dy = edge_end_lat - edge_start_lat

    if abs(dx) > abs(dy):
        edge_dir = 'WESTBOUND' if dx < 0 else 'EASTBOUND'
    else:
        edge_dir = 'SOUTHBOUND' if dy < 0 else 'NORTHBOUND'

    return tmc_dir.upper() == edge_dir

def find_matching_edges(tmc_df, sumo_edges, threshold=500.0):
    matches = []
    for _, row in tmc_df.iterrows():
        tmc_start_lat, tmc_start_lon = row['start_latitude'], row['start_longitude']
        tmc_end_lat, tmc_end_lon = row['end_latitude'], row['end_longitude']
        tmc_direction = row['direction']
        best_match = None
        min_dist = float('inf')

        for edge in sumo_edges:
            edge_start_lat, edge_start_lon = edge['start_lat'], edge['start_lon']
            edge_end_lat, edge_end_lon = edge['end_lat'], edge['end_lon']

            # Check same direction
            dist_start = distance(tmc_start_lat, tmc_start_lon, edge_start_lat, edge_start_lon)
            dist_end = distance(tmc_end_lat, tmc_end_lon, edge_end_lat, edge_end_lon)
            total_same = dist_start + dist_end
            if total_same < threshold and is_direction_match(tmc_direction, edge_start_lat, edge_start_lon, edge_end_lat, edge_end_lon):
                if total_same < min_dist:
                    min_dist = total_same
                    best_match = edge

            # Check reverse direction
            dist_start_rev = distance(tmc_start_lat, tmc_start_lon, edge_end_lat, edge_end_lon)
            dist_end_rev = distance(tmc_end_lat, tmc_end_lon, edge_start_lat, edge_start_lon)
            total_rev = dist_start_rev + dist_end_rev
            if total_rev < threshold and is_direction_match(tmc_direction, edge_end_lat, edge_end_lon, edge_start_lat, edge_start_lon):
                if total_rev < min_dist:
                    min_dist = total_rev
                    best_match = edge

        if best_match:
            matches.append({
                'tmc': row['tmc'],
                'sumo_edge_id': best_match['id'],
                'distance_diff': min_dist,
                'tmc_length_miles': row['miles'],
                'sumo_edge_length_m': best_match['length']
            })
        else:
            matches.append({
                'tmc': row['tmc'],
                'sumo_edge_id': None,
                'distance_diff': None,
                'tmc_length_miles': row['miles'],
                'sumo_edge_length_m': None
            })

    return pd.DataFrame(matches)

def main():
    sumo_net_file = 'roosevelt.net.xml'
    tmc_csv = 'data/TMC-03082025/TMC_Identification.csv'

    sumo_edges = parse_sumo_net(sumo_net_file)
    tmc_df = parse_tmc_data(tmc_csv)

    matches_df = find_matching_edges(tmc_df, sumo_edges, 50.0)
    print(matches_df)

if __name__ == '__main__':
    
    main()


