import streamlit as st
import time
import pandas as pd
import numpy as np
import json
import os
import math
from datetime import datetime, timedelta
from collections import deque
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union, cascaded_union

st.set_page_config(page_title="无人机地面站", layout="wide")

OBSTACLE_FILE = "obstacle_config.json"

def init_global_state():
    if "heartbeat_log" not in st.session_state:
        st.session_state.heartbeat_log = deque(maxlen=60)
    if "seq" not in st.session_state:
        st.session_state.seq = 0
    if "last_receive_time" not in st.session_state:
        st.session_state.last_receive_time = time.time()
    if "is_running" not in st.session_state:
        st.session_state.is_running = True
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "正常"
    if "last_heartbeat_time" not in st.session_state:
        st.session_state.last_heartbeat_time = 0
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "航线规划"
    if "coordinate_system" not in st.session_state:
        st.session_state.coordinate_system = "GCJ-02(高德/百度)"
    
    if "point_a" not in st.session_state:
        st.session_state.point_a = (32.234000, 118.743600)
    if "point_b" not in st.session_state:
        st.session_state.point_b = (32.238300, 118.745000)
    if "fly_height" not in st.session_state:
        st.session_state.fly_height = 50
    if "safety_radius" not in st.session_state:
        st.session_state.safety_radius = 5
    if "obstacles" not in st.session_state:
        st.session_state.obstacles = load_obstacles_from_file()
    if "point_a_set" not in st.session_state:
        st.session_state.point_a_set = False
    if "point_b_set" not in st.session_state:
        st.session_state.point_b_set = False
    
    if "pending_obstacle" not in st.session_state:
        st.session_state.pending_obstacle = None
    if "pending_obstacle_height" not in st.session_state:
        st.session_state.pending_obstacle_height = 30
    
    if "routes" not in st.session_state:
        st.session_state.routes = {"straight": [], "left": [], "right": [], "best": []}
    if "selected_route" not in st.session_state:
        st.session_state.selected_route = "best"
    
    if "mission_status" not in st.session_state:
        st.session_state.mission_status = "未开始"
    if "mission_progress" not in st.session_state:
        st.session_state.mission_progress = 0.0
    if "current_position" not in st.session_state:
        st.session_state.current_position = None
    if "mission_start_time" not in st.session_state:
        st.session_state.mission_start_time = None
    if "battery" not in st.session_state:
        st.session_state.battery = 100.0
    if "flight_log" not in st.session_state:
        st.session_state.flight_log = deque(maxlen=50)

def load_obstacles_from_file():
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("obstacles", [])
        except Exception as e:
            st.warning(f"加载障碍物配置失败: {str(e)}")
            return []
    return []

def save_obstacles_to_file(obstacles):
    try:
        with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"obstacles": obstacles}, f, indent=2)
        return True
    except Exception as e:
        st.error(f"保存障碍物配置失败: {str(e)}")
        return False

def gcj_to_wgs(lat, lon):
    pi = 3.14159265358979323846
    a = 6378137.0
    ee = 0.00669342162296594323
    
    dLat = transform_lat(lon - 105.0, lat - 35.0)
    dLon = transform_lon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * pi
    magic = np.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = np.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * np.cos(radLat) * pi)
    return lat - dLat, lon - dLon

def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret

def wgs_to_gcj(lat, lon):
    pi = 3.14159265358979323846
    a = 6378137.0
    ee = 0.00669342162296594323
    
    if out_of_china(lat, lon):
        return lat, lon
    
    dLat = transform_lat(lon - 105.0, lat - 35.0)
    dLon = transform_lon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * pi
    magic = np.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = np.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * np.cos(radLat) * pi)
    
    return lat + dLat, lon + dLon

def out_of_china(lat, lon):
    return not (73.66 <= lon <= 135.05 and 3.86 <= lat <= 53.55)

def deg_to_meters(lat, lon):
    lat_per_meter = 1 / 111111.0
    lon_per_meter = 1 / (111111.0 * math.cos(lat * math.pi / 180.0))
    return lat_per_meter, lon_per_meter

def generate_obstacle_buffers(obstacles, safety_radius):
    buffers = []
    for obs in obstacles:
        coords = obs.get("coordinates", [])
        if len(coords) >= 3:
            polygon = Polygon(coords)
            buffers.append(polygon)
            
            if safety_radius > 0:
                lat_per_meter, lon_per_meter = deg_to_meters(coords[0][0], coords[0][1])
                buffer_distance = safety_radius * lon_per_meter
                buffered = polygon.buffer(buffer_distance)
                buffers.append(buffered)
    return buffers

def can_fly_straight(point_a, point_b, buffers, fly_height, obstacles):
    for obs in obstacles:
        if obs.get("height", 0) >= fly_height:
            line = LineString([point_a, point_b])
            for buf in buffers:
                if line.intersects(buf):
                    return False
    return True

def calculate_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_routes(point_a, point_b, obstacles, safety_radius, fly_height):
    routes = {"straight": [], "left": [], "right": [], "best": []}
    
    buffers = generate_obstacle_buffers(obstacles, safety_radius)
    
    if can_fly_straight(point_a, point_b, buffers, fly_height, obstacles):
        routes["straight"] = [point_a, point_b]
        routes["best"] = [point_a, point_b]
    else:
        routes["straight"] = []
    
    all_points = []
    for obs in obstacles:
        all_points.extend(obs.get("coordinates", []))
    
    if all_points:
        lat_min = min(p[0] for p in all_points)
        lat_max = max(p[0] for p in all_points)
        lon_min = min(p[1] for p in all_points)
        lon_max = max(p[1] for p in all_points)
        
        lat_per_meter, lon_per_meter = deg_to_meters((lat_min + lat_max)/2, (lon_min + lon_max)/2)
        offset_distance = safety_radius * 3 * lon_per_meter
        
        mid_lat = (point_a[0] + point_b[0]) / 2
        mid_lon = (point_a[1] + point_b[1]) / 2
        
        left_wp = (mid_lat, lon_min - offset_distance)
        right_wp = (mid_lat, lon_max + offset_distance)
        
        routes["left"] = [point_a, left_wp, point_b]
        routes["right"] = [point_a, right_wp, point_b]
        
        dist_left = calculate_distance(point_a, left_wp) + calculate_distance(left_wp, point_b)
        dist_right = calculate_distance(point_a, right_wp) + calculate_distance(right_wp, point_b)
        
        if dist_left <= dist_right:
            routes["best"] = routes["left"]
        else:
            routes["best"] = routes["right"]
    else:
        routes["left"] = [point_a, point_b]
        routes["right"] = [point_a, point_b]
        routes["best"] = [point_a, point_b]
    
    return routes

def create_flight_map(show_routes=True, show_drone=False, drone_pos=None):
    if st.session_state.point_a_set and st.session_state.point_b_set:
        center_lat = (st.session_state.point_a[0] + st.session_state.point_b[0]) / 2
        center_lon = (st.session_state.point_a[1] + st.session_state.point_b[1]) / 2
    elif st.session_state.point_a_set:
        center_lat, center_lon = st.session_state.point_a
    else:
        center_lat, center_lon = (32.23615, 118.7443)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )
    
    if st.session_state.point_a_set:
        folium.CircleMarker(
            location=st.session_state.point_a,
            radius=8,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8,
            popup=folium.Popup(f"起点A\n纬度: {st.session_state.point_a[0]:.6f}\n经度: {st.session_state.point_a[1]:.6f}", max_width=300)
        ).add_to(m)
    
    if st.session_state.point_b_set:
        folium.CircleMarker(
            location=st.session_state.point_b,
            radius=8,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.8,
            popup=folium.Popup(f"终点B\n纬度: {st.session_state.point_b[0]:.6f}\n经度: {st.session_state.point_b[1]:.6f}", max_width=300)
        ).add_to(m)
    
    if show_routes and st.session_state.routes:
        routes = st.session_state.routes
        selected = st.session_state.selected_route
        
        if routes["straight"]:
            folium.PolyLine(
                locations=routes["straight"],
                color='green' if selected == "straight" else '#90EE90',
                weight=5 if selected == "straight" else 2,
                opacity=0.8 if selected == "straight" else 0.4,
                popup="直飞航线"
            ).add_to(m)
        
        if routes["left"]:
            folium.PolyLine(
                locations=routes["left"],
                color='blue' if selected == "left" else '#87CEEB',
                weight=5 if selected == "left" else 2,
                opacity=0.8 if selected == "left" else 0.4,
                popup="左绕航线"
            ).add_to(m)
        
        if routes["right"]:
            folium.PolyLine(
                locations=routes["right"],
                color='orange' if selected == "right" else '#FFA500',
                weight=5 if selected == "right" else 2,
                opacity=0.8 if selected == "right" else 0.4,
                popup="右绕航线"
            ).add_to(m)
        
        if routes["best"] and selected == "best":
            folium.PolyLine(
                locations=routes["best"],
                color='yellow',
                weight=6,
                opacity=0.9,
                popup="最佳航线"
            ).add_to(m)
    
    for i, obstacle in enumerate(st.session_state.obstacles):
        if "coordinates" in obstacle:
            coords = obstacle["coordinates"]
            if isinstance(coords[0], list):
                coords = [(p[0], p[1]) for p in coords]
            height = obstacle.get("height", "未设置")
            folium.Polygon(
                locations=coords,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.4,
                weight=2,
                popup=folium.Popup(f"障碍物 {i+1}\n高度: {height}m\n顶点数: {len(coords)}", max_width=300)
            ).add_to(m)
    
    if show_drone and drone_pos:
        folium.CircleMarker(
            location=drone_pos,
            radius=10,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1.0,
            popup="无人机"
        ).add_to(m)
        
        if st.session_state.routes and st.session_state.routes[st.session_state.selected_route]:
            route = st.session_state.routes[st.session_state.selected_route]
            progress = st.session_state.mission_progress
            flown_points = interpolate_route(route, progress)
            if len(flown_points) >= 2:
                folium.PolyLine(
                    locations=flown_points,
                    color='blue',
                    weight=4,
                    opacity=0.8,
                    popup="已飞轨迹"
                ).add_to(m)
    
    Draw(
        draw_options={
            'polyline': False,
            'rectangle': False,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={
            'edit': True
        }
    ).add_to(m)
    
    return m

def interpolate_route(route, progress):
    if not route or len(route) < 2:
        return []
    
    total_dist = 0
    segments = []
    for i in range(len(route) - 1):
        dist = calculate_distance(route[i], route[i+1])
        segments.append((route[i], route[i+1], dist))
        total_dist += dist
    
    target_dist = total_dist * progress
    current_dist = 0
    
    for start, end, seg_dist in segments:
        if current_dist + seg_dist >= target_dist:
            ratio = (target_dist - current_dist) / seg_dist
            lat = start[0] + (end[0] - start[0]) * ratio
            lon = start[1] + (end[1] - start[1]) * ratio
            result = [start]
            for i in range(len(route)):
                if route[i] == start:
                    result.append((lat, lon))
                    break
                result.append(route[i])
            return result
        current_dist += seg_dist
    
    return route.copy()

def get_current_position(route, progress):
    if not route or len(route) < 2:
        return None
    
    total_dist = 0
    segments = []
    for i in range(len(route) - 1):
        dist = calculate_distance(route[i], route[i+1])
        segments.append((route[i], route[i+1], dist))
        total_dist += dist
    
    if total_dist == 0:
        return route[0]
    
    target_dist = total_dist * progress
    current_dist = 0
    
    for start, end, seg_dist in segments:
        if current_dist + seg_dist >= target_dist:
            ratio = (target_dist - current_dist) / seg_dist
            lat = start[0] + (end[0] - start[0]) * ratio
            lon = start[1] + (end[1] - start[1]) * ratio
            return (lat, lon)
        current_dist += seg_dist
    
    return route[-1]

def update_heartbeat():
    if not st.session_state.is_running:
        return
    
    now = time.time()
    if now - st.session_state.last_heartbeat_time >= 1.0:
        st.session_state.seq += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.heartbeat_log.append({
            "序号": st.session_state.seq,
            "时间": timestamp,
            "状态": "正常"
        })
        st.session_state.last_heartbeat_time = now
        st.session_state.last_receive_time = now
    
    if now - st.session_state.last_receive_time > 3.0:
        st.session_state.connection_status = "超时"
    else:
        st.session_state.connection_status = "正常"

def reset_heartbeat_data():
    st.session_state.heartbeat_log.clear()
    st.session_state.seq = 0
    st.session_state.last_receive_time = time.time()
    st.session_state.last_heartbeat_time = 0
    st.session_state.connection_status = "正常"

def heartbeat_monitor():
    st.title("✈️ 无人机飞行监控")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.session_state.is_running = st.checkbox("启动心跳模拟", value=st.session_state.is_running)
        if st.button("重置数据"):
            reset_heartbeat_data()
    
    update_heartbeat()
    
    if st.session_state.connection_status == "超时":
        st.error("🚨 连接超时！无人机已超过3秒未发送心跳包")
    else:
        st.success("✅ 心跳连接正常")
    
    df = pd.DataFrame(st.session_state.heartbeat_log) if st.session_state.heartbeat_log else pd.DataFrame(columns=["序号", "时间", "状态"])
    
    st.subheader("📋 心跳包记录（最近60条）")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("📈 心跳序号趋势图")
    if not df.empty:
        chart_data = df.set_index("时间")["序号"]
        st.line_chart(chart_data, use_container_width=True)
    else:
        st.info("暂无数据，请等待心跳包生成...")

def load_default_obstacles():
    default_obstacles = [
        {
            "coordinates": [
                (32.2355, 118.7440),
                (32.2353, 118.7445),
                (32.2356, 118.7447),
                (32.2358, 118.7442)
            ],
            "height": 40
        },
        {
            "coordinates": [
                (32.2360, 118.7438),
                (32.2358, 118.7443),
                (32.2361, 118.7445),
                (32.2363, 118.7440)
            ],
            "height": 55
        },
        {
            "coordinates": [
                (32.2365, 118.7442),
                (32.2363, 118.7448),
                (32.2366, 118.7450),
                (32.2368, 118.7444)
            ],
            "height": 35
        }
    ]
    st.session_state.obstacles = default_obstacles
    save_obstacles_to_file(default_obstacles)
    st.success("已加载默认障碍物数据")

def flight_map_page():
    st.title("🌍 无人机航线规划")
    
    if "pending_obstacle" not in st.session_state:
        st.session_state.pending_obstacle = None
    if "pending_obstacle_height" not in st.session_state:
        st.session_state.pending_obstacle_height = 30
    
    st.subheader("📍 飞行地图")
    m = create_flight_map()
    output = st_folium(m, width=900, height=600)
    
    if output and 'last_active_drawing' in output:
        last_drawing = output['last_active_drawing']
        if last_drawing:
            geometry = last_drawing.get('geometry', {})
            if geometry.get('type') == 'Polygon':
                coordinates = geometry.get('coordinates', [])
                if coordinates:
                    coords = [(p[1], p[0]) for p in coordinates[0]]
                    st.session_state.pending_obstacle = coords
                    st.info("已绘制新障碍物，请设置高度后保存")
    
    if st.session_state.pending_obstacle is not None:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.session_state.pending_obstacle_height = st.number_input(
                "障碍物高度(m)", 
                value=st.session_state.pending_obstacle_height, 
                min_value=1, 
                max_value=500, 
                step=1
            )
        with col2:
            if st.button("保存障碍物"):
                exists = False
                for obs in st.session_state.obstacles:
                    if obs.get("coordinates") == st.session_state.pending_obstacle:
                        exists = True
                        break
                if not exists:
                    st.session_state.obstacles.append({
                        "coordinates": st.session_state.pending_obstacle,
                        "height": st.session_state.pending_obstacle_height
                    })
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.success(f"障碍物已保存，高度: {st.session_state.pending_obstacle_height}m")
                else:
                    st.warning("该障碍物已存在")
                st.session_state.pending_obstacle = None
    
    st.sidebar.markdown("### 起点A")
    a_lat = st.sidebar.number_input("纬度", value=st.session_state.point_a[0], step=0.0001, format="%.6f")
    a_lon = st.sidebar.number_input("经度", value=st.session_state.point_a[1], step=0.0001, format="%.6f")
    if st.sidebar.button("设置A点"):
        if st.session_state.coordinate_system == "WGS-84":
            gcj_lat, gcj_lon = wgs_to_gcj(a_lat, a_lon)
            st.session_state.point_a = (gcj_lat, gcj_lon)
        else:
            st.session_state.point_a = (a_lat, a_lon)
        st.session_state.point_a_set = True
        st.sidebar.success("A点已设置")
    
    st.sidebar.markdown("### 终点B")
    b_lat = st.sidebar.number_input("纬度", value=st.session_state.point_b[0], step=0.0001, format="%.6f", key="b_lat")
    b_lon = st.sidebar.number_input("经度", value=st.session_state.point_b[1], step=0.0001, format="%.6f", key="b_lon")
    if st.sidebar.button("设置B点"):
        if st.session_state.coordinate_system == "WGS-84":
            gcj_lat, gcj_lon = wgs_to_gcj(b_lat, b_lon)
            st.session_state.point_b = (gcj_lat, gcj_lon)
        else:
            st.session_state.point_b = (b_lat, b_lon)
        st.session_state.point_b_set = True
        st.sidebar.success("B点已设置")
    
    st.sidebar.markdown("### 飞行参数")
    st.session_state.fly_height = st.sidebar.number_input("飞行高度(m)", value=st.session_state.fly_height, min_value=1, max_value=500, step=1)
    st.session_state.safety_radius = st.sidebar.number_input("安全半径(m)", value=st.session_state.safety_radius, min_value=1, max_value=50, step=1)
    
    if st.sidebar.button("生成航线"):
        if st.session_state.point_a_set and st.session_state.point_b_set:
            routes = generate_routes(
                st.session_state.point_a,
                st.session_state.point_b,
                st.session_state.obstacles,
                st.session_state.safety_radius,
                st.session_state.fly_height
            )
            st.session_state.routes = routes
            st.sidebar.success("航线生成完成")
            
            if routes["straight"]:
                st.sidebar.info("✓ 可直飞")
            else:
                st.sidebar.info("✗ 需绕行")
        else:
            st.sidebar.warning("请先设置A点和B点")
    
    st.sidebar.markdown("### 航线选择")
    route_options = ["best", "straight", "left", "right"]
    route_labels = {"best": "最佳航线", "straight": "直飞航线", "left": "左绕航线", "right": "右绕航线"}
    st.session_state.selected_route = st.sidebar.radio(
        "选择航线",
        route_options,
        format_func=lambda x: route_labels[x],
        index=route_options.index(st.session_state.selected_route)
    )
    
    st.sidebar.markdown("### 障碍物管理")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("保存到文件"):
            if save_obstacles_to_file(st.session_state.obstacles):
                st.success(f"已保存 {len(st.session_state.obstacles)} 个障碍物")
    
    with col2:
        if st.button("从文件加载"):
            obstacles = load_obstacles_from_file()
            st.session_state.obstacles = obstacles
            st.success(f"已加载 {len(obstacles)} 个障碍物")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("清除全部"):
            st.session_state.obstacles = []
            st.session_state.routes = {"straight": [], "left": [], "right": [], "best": []}
            st.success("已清除所有障碍物")
    
    with col4:
        if st.button("一键部署"):
            load_default_obstacles()
    
    if st.session_state.obstacles:
        st.subheader("📋 障碍物列表")
        obs_data = []
        for i, obs in enumerate(st.session_state.obstacles):
            coords = obs.get("coordinates", [])
            height = obs.get("height", "未设置")
            if len(coords) >= 2:
                avg_lat = sum(p[0] for p in coords) / len(coords)
                avg_lon = sum(p[1] for p in coords) / len(coords)
            else:
                avg_lat, avg_lon = 0, 0
            obs_data.append({
                "编号": i + 1,
                "顶点数": len(coords),
                "中心纬度": f"{avg_lat:.6f}",
                "中心经度": f"{avg_lon:.6f}",
                "高度(m)": height
            })
        obs_df = pd.DataFrame(obs_data)
        st.dataframe(obs_df, use_container_width=True)

def flight_monitor_page():
    st.title("🛫 飞行监控")
    
    flight_speed = 8.5
    
    def add_log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.flight_log.append(f"[{timestamp}] {message}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("开始任务", disabled=(st.session_state.mission_status == "运行中")):
            if st.session_state.point_a_set and st.session_state.point_b_set:
                if not st.session_state.routes or not st.session_state.routes[st.session_state.selected_route]:
                    routes = generate_routes(
                        st.session_state.point_a,
                        st.session_state.point_b,
                        st.session_state.obstacles,
                        st.session_state.safety_radius,
                        st.session_state.fly_height
                    )
                    st.session_state.routes = routes
                
                if st.session_state.routes[st.session_state.selected_route]:
                    st.session_state.mission_status = "运行中"
                    st.session_state.mission_progress = 0.0
                    st.session_state.mission_start_time = time.time()
                    st.session_state.battery = 100.0
                    st.session_state.flight_log.clear()
                    add_log("任务开始")
                    add_log(f"航线: {{'best': '最佳', 'straight': '直飞', 'left': '左绕', 'right': '右绕'}}[st.session_state.selected_route]")
                else:
                    st.warning("无法生成有效航线")
            else:
                st.warning("请先在航线规划页面设置A点和B点")
    
    with col2:
        if st.button("暂停", disabled=(st.session_state.mission_status != "运行中")):
            st.session_state.mission_status = "暂停"
            add_log("任务暂停")
    
    with col3:
        if st.button("停止", disabled=(st.session_state.mission_status not in ["运行中", "暂停"])):
            st.session_state.mission_status = "未开始"
            st.session_state.mission_progress = 0.0
            st.session_state.mission_start_time = None
            st.session_state.current_position = None
            add_log("任务停止")
    
    st.subheader("📊 状态指示")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 GCS在线")
    with col2:
        st.success("🟢 OBC在线")
    with col3:
        st.success("🟢 FCU在线")
    
    route = st.session_state.routes.get(st.session_state.selected_route, [])
    total_distance = 0
    if len(route) >= 2:
        for i in range(len(route) - 1):
            total_distance += calculate_distance(route[i], route[i+1])
    
    current_distance = total_distance * st.session_state.mission_progress
    remaining_distance = total_distance - current_distance
    
    if st.session_state.mission_status == "运行中" and st.session_state.mission_start_time:
        elapsed_time = time.time() - st.session_state.mission_start_time
        st.session_state.mission_progress = min(1.0, (elapsed_time * flight_speed) / max(total_distance, 1))
        
        if st.session_state.mission_progress >= 1.0:
            st.session_state.mission_status = "完成"
            st.session_state.mission_progress = 1.0
            add_log("任务完成")
        
        battery_consumption = (current_distance / 1000) * 1
        st.session_state.battery = max(0.0, 100.0 - battery_consumption)
    
    st.subheader("📈 任务信息")
    col1, col2 = st.columns([1, 3])
    with col1:
        total_waypoints = len(route)
        current_waypoint = 0
        if st.session_state.mission_progress > 0:
            current_waypoint = min(total_waypoints, int(st.session_state.mission_progress * total_waypoints) + 1)
        
        st.metric("当前航点", f"{current_waypoint} / {total_waypoints}")
        st.metric("飞行速度", f"{flight_speed} m/s")
        st.metric("剩余距离", f"{remaining_distance/1000:.2f} km")
        
        if remaining_distance > 0:
            eta_seconds = remaining_distance / flight_speed
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            st.metric("预计到达", eta_str)
        else:
            st.metric("预计到达", "已到达")
        
        st.progress(st.session_state.mission_progress)
        st.write(f"任务进度: {st.session_state.mission_progress:.1%}")
    
    with col2:
        st.subheader("⚡ 电量")
        st.progress(st.session_state.battery / 100)
        st.write(f"剩余电量: {st.session_state.battery:.1f}%")
    
    st.subheader("🗺️ 实时飞行地图")
    current_pos = get_current_position(route, st.session_state.mission_progress)
    m = create_flight_map(show_routes=True, show_drone=True, drone_pos=current_pos)
    st_folium(m, width=900, height=500)
    
    st.subheader("🔗 通信链路拓扑")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### GCS")
        st.markdown("**地面控制站**")
    with col2:
        st.markdown("### OBC")
        st.markdown("**机载计算机**")
    with col3:
        st.markdown("### FCU")
        st.markdown("**飞行控制单元**")
    
    st.markdown("---")
    st.markdown("**通信状态:**")
    st.markdown("🟢 GCS ↔ OBC: 正常")
    st.markdown("🟢 OBC ↔ FCU: 正常")
    
    st.subheader("📝 飞行日志")
    for log in reversed(st.session_state.flight_log):
        st.write(log)
    
    if st.session_state.mission_status == "运行中":
        time.sleep(0.5)
        st.rerun()

def main():
    init_global_state()
    
    with st.sidebar:
        st.header("功能页面")
        st.session_state.current_page = st.radio(
            "选择页面",
            ["航线规划", "飞行监控"],
            index=["航线规划", "飞行监控"].index(st.session_state.current_page),
            horizontal=False
        )
        
        st.divider()
        
        st.subheader("坐标系设置")
        st.session_state.coordinate_system = st.selectbox(
            "输入坐标系",
            ["GCJ-02(高德/百度)", "WGS-84"],
            index=["GCJ-02(高德/百度)", "WGS-84"].index(st.session_state.coordinate_system)
        )
        
        st.divider()
        
        st.subheader("系统状态")
        a_status = "已设" if st.session_state.point_a_set else "未设"
        b_status = "已设" if st.session_state.point_b_set else "未设"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.point_a_set:
                st.success(f"A点 {a_status}")
            else:
                st.info(f"A点 {a_status}")
        with col2:
            if st.session_state.point_b_set:
                st.success(f"B点 {b_status}")
            else:
                st.info(f"B点 {b_status}")
    
    if st.session_state.current_page == "航线规划":
        flight_map_page()
    else:
        flight_monitor_page()

if __name__ == "__main__":
    main()