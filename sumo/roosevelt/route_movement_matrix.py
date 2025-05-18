"""
From Ziyi Zhang
"""
import pandas as pd
import xml.etree.ElementTree as ET
from sumolib.net import readNet


# ======== Step 1: 读取 SUMO 网络文件 ========
net = readNet("roosevelt.net.xml")


# ======== Step 2: 解析所有 movement (from_edge -> to_edge) ========
movement_list = []
movement_index_map = {}


for edge in net.getEdges():
    from_edge = edge.getID()
    for conn in edge.getOutgoing():
        to_edge = conn.getID()
        movement_id = f"{from_edge}->{to_edge}"
        if movement_id not in movement_index_map:
            movement_index_map[movement_id] = len(movement_list)
            movement_list.append((from_edge, to_edge))




# ======== Step 3: 解析 .rou.xml 中的 route ========
def parse_routes_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    route_dict = {}
    for route in root.iter("route"):
        route_id = route.attrib.get("id")
        edges = route.attrib.get("edges")
        if route_id and edges:
            edge_list = edges.strip().split()
            route_dict[route_id] = edge_list
    return route_dict


route_dict = parse_routes_from_xml("roosevelt.rou.xml")
route_list = list(route_dict.keys())


# ======== Step 4: 构建 Route-Movement Matrix ========
A = pd.DataFrame(0, index=movement_index_map.keys(), columns=route_list)


for route_id, edge_seq in route_dict.items():
    for i in range(len(edge_seq) - 1):
        from_edge = edge_seq[i]
        to_edge = edge_seq[i + 1]
        movement_id = f"{from_edge}->{to_edge}"
        if movement_id in A.index:
            A.loc[movement_id, route_id] = 1


# ======== Step 5: 输出结果 ========
print("Route-Movement Matrix A[l, r]:")
print(A.head())
print("A size: ", A.shape)


# 可保存成 csv 方便查看
A.to_csv("data/route_movement_matrix.csv")
