import mujoco as mj
import numpy as np
import time
import os
import glob
import numpy.typing as npt
from typing import Sequence
import matplotlib.pyplot as plt
import imageio.v3 as iio
from mujoco import viewer

# ========== model init ==========
model_path = "../models/pm_v2/xml/serial_pm_v2.xml"
model = mj.MjModel.from_xml_path(model_path)
data = mj.MjData(model)

joint_adr=[
"0 None t           ype= 0  qpos_adr= 0 dof_adr= 0",
"1 J00_HIP_PITCH_L  type= 3 qpos_adr= 7 dof_adr= 6",
"2 J01_HIP_ROLL_L   type= 3 qpos_adr= 8 dof_adr= 7",
"3 J02_HIP_YAW_L    type= 3 qpos_adr= 9 dof_adr= 8",
"4 J03_KNEE_PITCH_L type= 3 qpos_adr= 10 dof_adr= 9",
"5 J04_ANKLE_PITCH_L type= 3 qpos_adr= 11 dof_adr= 10",
"6 J05_ANKLE_ROLL_L type= 3 qpos_adr= 12 dof_adr= 11",
"7 J06_HIP_PITCH_R  type= 3 qpos_adr= 13 dof_adr= 12",
"8 J07_HIP_ROLL_R   type= 3 qpos_adr= 14 dof_adr= 13",
"9 J08_HIP_YAW_R    type= 3 qpos_adr= 15 dof_adr= 14",
"10 J09_KNEE_PITCH_R type= 3 qpos_adr= 16 dof_adr= 15",
"11 J10_ANKLE_PITCH_R type= 3 qpos_adr= 17 dof_adr= 16",
"12 J11_ANKLE_ROLL_R type= 3 qpos_adr= 18 dof_adr= 17",
"13 J12_WAIST_YAW   type= 3 qpos_adr= 19 dof_adr= 18",
"14 J13_SHOULDER_PITCH_L type= 3 qpos_adr= 20 dof_adr= 19"
"15 J14_SHOULDER_ROLL_L type= 3 qpos_adr= 21 dof_adr= 20",
"16 J15_SHOULDER_YAW_L  type= 3 qpos_adr= 22 dof_adr= 21",
"17 J16_ELBOW_PITCH_L   type= 3 qpos_adr= 23 dof_adr= 22"
"18 J17_ELBOW_YAW_L     type= 3 qpos_adr= 24 dof_adr= 23",
"19 J18_SHOULDER_PITCH_R type= 3 qpos_adr= 25 dof_adr= 24",
"20 J19_SHOULDER_ROLL_R type= 3 qpos_adr= 26 dof_adr= 25",
"21 J20_SHOULDER_YAW_R  type= 3 qpos_adr= 27 dof_adr= 26",
"22 J21_ELBOW_PITCH_R   type= 3 qpos_adr= 28 dof_adr= 27",
"23 J22_ELBOW_YAW_R     type= 3 qpos_adr= 29 dof_adr= 28",
"24 J23_HEAD_YAW        type= 3 qpos_adr= 30 dof_adr= 29,"]
joint_names = [
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_YAW"
]
# joint_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# qadr_arr =  [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# dadr_arr =  [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
qadr_arr = [int(model.jnt_qposadr[j]) for j in joint_ids]   # 每个关节在 qpos 的槽位
dadr_arr = [int(model.jnt_dofadr[j])  for j in joint_ids]   # 每个关节在 qvel/qfrc_applied 的槽位
# ========== model init end ==========


kp = [  200.0, 200.0, 380.0, 450.0, 400.0, 200.0, 
        200.0, 200.0, 380.0, 450.0, 400.0, 200.0, 
        200.0,
        250.0, 250.0, 250.0, 250.0, 250.0,  
        250.0, 250.0, 250.0, 250.0, 250.0, 
        100.0]
kd = [  5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 
        5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 
        1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0]


# 目标向量：可以是 24 维，也可以是标量（标量会自动广播）
q_target_1: np.array = [    
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
    -0.131, 0.144, -0.41, -1.57, -0.513,
 	-0.131, -0.144, 0.41, -1.57, -0.513,
    0   # shake two hands
] 
q_target_2: np.array = [    
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
    -1.5, 0, 0.04, 0.04, 0.03,    
 	-1.5, 0, -0.02, -0.04, -0.002,   
    0
]
q_target_3: np.array = [     
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
 	-1, -0.3565, 0, -1.5, 0.5,
 	-1, 0.4446, 0, -1.5, -0.5, 
    0]
q_target_4: np.array = [     
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
 	-0.2432, 0.7459, -0.1974, -1.9934, 0.8905,
 	-0.07992, -2.1483, -0.03414, -0.01354, -2.0476, 
    0]
q_target_5: np.array = [    
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
    -0.04406, -0.08869, 0.4644, -0.1165, 0.8905,
    -0.20466, -1.3983, -1.2522, -2.1879, -1.46315,
    0] 
q_target_6: np.array = [     
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0,
    0, 
    -2, 0, -0.04, 0.3, -0.03,    
    -2,  0, 0.02, 0.3, 0.002,  
    0]
q_target_total: np.array = [q_target_1,
                            q_target_2,
                            q_target_3,
                            q_target_4,
                            q_target_5,
                            q_target_6]

root_pos0  = data.qpos[0:3].copy()
root_quat0 = data.qpos[3:7].copy()
def pin_root(model, data):
    data.qpos[0:3] = root_pos0
    data.qpos[3:7] = root_quat0
    data.qvel[0:6] = 0.0
    mj.mj_forward(model, data)


def computeLinear(t,q_target):
    if ramp > 1e-6:
        k = np.clip(t / ramp, 0.0, 1.0)
    else:
        k = 1.0
    return q0 + k * (q_target - q0)   # 返回 24 维目标


def compute_joint_traj_five_poly(time_lim: float,
                                 pos_start: Sequence[float],
                                 pos_end: Sequence[float],
                                 sample_rate: float
                                 ):
    pos_start = np.asarray(pos_start, dtype=float)
    pos_end   = np.asarray(pos_end,   dtype=float)
    pos_dist = np.array(pos_end) - pos_start

    # 位置多项式系数：q(t) = a0(=pos_start) + a1(=0)*t + a2(=0)*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    t3_coef = (10.0 * pos_dist) / (time_lim ** 3)
    t4_coef = (-15.0 * pos_dist) / (time_lim ** 4)
    t5_coef = (6.0 * pos_dist) / (time_lim ** 5)

    # 速度多项式系数：q'(t) = a1(=0) + 2*a2(=0)*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
    v2_coef = 3.0 * t3_coef
    v3_coef = 4.0 * t4_coef
    v4_coef = 5.0 * t5_coef

    pos_coef = np.array([pos_start, [0.0]*24, [0.0]*24, t3_coef, t4_coef, t5_coef])
    vel_coef = np.array([[0.0]*24, [0.0]*24, v2_coef, v3_coef, v4_coef])
    
    #pos_coef.shape = 24 x 6, vel_coef.shape = 24 x 5
    return pos_coef, vel_coef

# auto save coef for multiple targets
# the following commented code is an example
'''        
    [p1_coef, v1_coef] = compute_joint_traj_five_poly(move_time, q0, q_target_1, sample_rate)
    [p2_coef, v2_coef] = compute_joint_traj_five_poly(move_time, q_target_1, q_target_2, sample_rate)
    [p3_coef, v3_coef] = compute_joint_traj_five_poly(move_time, q_target_2, q_target_3, sample_rate)
    [p4_coef, v4_coef] = compute_joint_traj_five_poly(move_time, q_target_3, q_target_4, sample_rate)
    [p5_coef, v5_coef] = compute_joint_traj_five_poly(move_time, q_target_4, q_target_5, sample_rate)
    [p6_coef, v6_coef] = compute_joint_traj_five_poly(move_time, q_target_5, q_target_6, sample_rate)

    p_coef = [p1_coef, p2_coef, p3_coef, p4_coef, p5_coef, p6_coef]
    v_coef = [v1_coef, v2_coef, v3_coef, v4_coef, v5_coef, v6_coef]  
'''

def save_coef(time_lim: float,
              pos_start: Sequence[float],
              sample_rate: float,
              q_target_total: Sequence[np.array]
              ):
    k = len(q_target_total)
    p_save = []
    v_save = []
    for i in range(k):
        if i == 0:
            [p_coef, v_coef] = compute_joint_traj_five_poly(time_lim, pos_start, q_target_total[i], sample_rate)
        else:
            [p_coef, v_coef] = compute_joint_traj_five_poly(time_lim, q_target_total[i-1], q_target_total[i], sample_rate)
        p_save.append(p_coef)
        v_save.append(v_coef)

    return p_save, v_save



# ========== sim init ==========
# simulation time step
model.opt.timestep = 0.01
sample_rate = 1.0 / model.opt.timestep
ramp = 0.5
move_time = 3.0
dof:int = 24
mj.mj_forward(model, data)

# read q0, dq0; len(q0) = 24, len(dq0) = 24
q0  = data.qpos[qadr_arr].copy()      # 读取时用 copy，避免后续被原地改动影响
dq0 = data.qvel[dadr_arr].copy()
q = np.array(dof)
dq = np.array(dof)
tau = np.array(dof)

p_coef = []
v_coef =[] 
p_coef, v_coef = save_coef(move_time, q0, sample_rate, q_target_total)

t_sim = 0.0
t_count = time.perf_counter()
# ========== sim init end ==========

# ========== main loop ==========
with viewer.launch_passive(model, data) as v:
    v.sync()
  
    # t0 is for time.sleep()
    t0 = time.perf_counter()

    while v.is_running() and t_sim < len(p_coef)*move_time + 1:
        
        # keep root fixed
        pin_root(model, data)

        # five order polynomial interpolation      
        for i in range(len(p_coef)):
            if i*move_time <= t_sim < (i+1)*move_time:
                t = t_sim - move_time*i
                t2 = t*t
                t3 = t2*t
                t4 = t3*t
                t5 = t4*t
                q_des = p_coef[i][0] + p_coef[i][1]*t + p_coef[i][2]*t2 + p_coef[i][3]*t3 + p_coef[i][4]*t4 + p_coef[i][5]*t5
                dq_des = v_coef[i][0] + v_coef[i][1]*t + v_coef[i][2]*t2 + v_coef[i][3]*t3 + v_coef[i][4]*t4

        q     = data.qpos[7:31]   # read current pos: 7 - 31 is the pos adress index of 24 joints
        dq    = data.qvel[6:30]  # read current vel: 6 - 30 is the vel adress index of 24 joints     
        tau   = kp * (q_des - q) - kd * dq

        # send torque to simulator
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dadr_arr] = tau

        # 推进一步并刷新画面
        mj.mj_step(model, data)
        v.sync()
        t_sim += model.opt.timestep
        
        t_count += model.opt.timestep
        time.sleep(max(0.0, t_count - time.perf_counter()))
        # print("time = ", t_sim)
