import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skyfield.api import Loader, utc

MU = 0.0121505856 

def stk_to_cr3bp_v2(stk_file_path, output_file_path):
    load = Loader('~/skyfield-data')
    ts = load.timescale()
    planets = load('de421.bsp')
    earth, moon = planets['earth'], planets['moon']

    data_lines = []
    epoch_str = ""
    dist_unit_scale = 1.0 
    central_body = "Earth" # 默认地心
    
    # 1. 增强版解析器：自动识别中心天体
    with open(stk_file_path, 'r') as f:
        for line in f:
            if "ScenarioEpoch" in line:
                epoch_str = " ".join(line.split()[1:5])
            if "CentralBody" in line:
                central_body = line.split()[-1] # 识别是 Earth 还是 Moon
            if "DistanceUnit" in line:
                if "Kilometers" in line: dist_unit_scale = 1000.0
            if "EphemerisTimePosVel" in line:
                break
        
        for line in f:
            if "END" in line: break
            parts = line.split()
            if len(parts) == 7:
                data_lines.append([float(x) for x in parts])

    print(f"检测到 STK 数据中心天体为: {central_body}")
    scenario_epoch = datetime.strptime(epoch_str, "%d %b %Y %H:%M:%S.%f").replace(tzinfo=utc)
    results = []

    for row in data_lines:
        t_sec = row[0]
        r_sat_stk = np.array(row[1:4]) * dist_unit_scale
        v_sat_stk = np.array(row[4:7]) * dist_unit_scale

        current_time = scenario_epoch + timedelta(seconds=t_sec)
        t_sky = ts.from_datetime(current_time)

        # 获取精确的地月矢量 (Earth -> Moon)
        em_state = earth.at(t_sky).observe(moon)
        r_em_i = em_obs = em_state.position.m 
        v_em_i = em_state.velocity.m_per_s

        L = np.linalg.norm(r_em_i)
        h_vec = np.cross(r_em_i, v_em_i)
        omega_mag = np.linalg.norm(h_vec) / (L**2) 
        
        # 旋转基矢量
        unit_i = r_em_i / L
        unit_k = h_vec / np.linalg.norm(h_vec)
        unit_j = np.cross(unit_k, unit_i)
        C_i2s = np.vstack([unit_i, unit_j, unit_k])

        # --- 核心修正：根据参考天体进行平移 ---
        if central_body.lower() == "earth":
            # 如果输入是相对于地球的：转为相对于质心
            r_rel_i = r_sat_stk - (MU * r_em_i)
            v_rel_i = v_sat_stk - (MU * v_em_i)
        elif central_body.lower() == "moon":
            # 如果输入是相对于月球的：转为相对于质心
            r_rel_i = r_sat_stk + ((1 - MU) * r_em_i)
            v_rel_i = v_sat_stk + ((1 - MU) * v_em_i)
        else:
            raise ValueError(f"暂不支持天体: {central_body}")

        # 旋转变换
        omega_vec = omega_mag * unit_k
        r_rot = C_i2s @ r_rel_i
        v_rot = C_i2s @ (v_rel_i - np.cross(omega_vec, r_rel_i))

        # 无量纲化
        r_nd = r_rot / L
        v_nd = v_rot / (L * omega_mag)
        t_nd = t_sec * omega_mag 

        results.append([t_nd, *r_nd, *v_nd])

    results = np.array(results)
    
    # 绘图逻辑优化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(results[:,1], results[:,2], results[:,3], label='Orbit')
    
    # 绘制地球和月球点
    ax.scatter(-MU, 0, 0, color='blue', s=100, label='Earth')
    ax.scatter(1-MU, 0, 0, color='red', s=50, label='Moon')
    
    # 强制放大到月球附近观察
    ax.set_xlim(1-MU-0.05, 1-MU+0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(-0.05, 0.05)
    
    ax.set_title(f"Zoomed view near Moon (Center: {central_body})")
    ax.legend()
    plt.show()

# 执行
stk_to_cr3bp_v2(r'C:\Users\11474\Documents\STK 11 (x64)\Scenario2\Satellite2.e', r'C:\Users\11474\Desktop\Pytorch\cr3bp_ephemeris2.txt')
 