import mujoco as mj
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import imageio.v3 as iio
from mujoco import viewer


model_path = "../models/pm_v2/xml/serial_pm_v2.xml"
model = mj.MjModel.from_xml_path(model_path)
data = mj.MjData(model)
# disable gravity
#model.opt.gravity[:] = 0.0
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
joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[15])

qadr = int(model.jnt_qposadr[joint_id])
dadr = int(model.jnt_dofadr[joint_id])


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


root_pos0  = data.qpos[0:3].copy()
root_quat0 = data.qpos[3:7].copy()
def pin_root(model, data):
    data.qpos[0:3] = root_pos0
    data.qpos[3:7] = root_quat0
    data.qvel[0:6] = 0.0
    mj.mj_forward(model, data)


def q_target(t):
    if t <= ramp and ramp > 1e-6:
        k = t / ramp
        return (1.0 - k) * q0 + k * q_target_final
    return q_target_final





# ========== sim init ==========
if joint_id < 0:
    raise RuntimeError("joint id not found")

# simulation time step
model.opt.timestep = 0.002 
mj.mj_forward(model, data)
ramp = 0.5
t_sim = 0.0
move_time = 3.0

# target traj
q0 = float(data.qpos[qadr])
q_target_final = 0.3


# ========== sim init end ==========

# print joint info
'''
types = model.jnt_type  # 0=free, 1=ball, 2=slide, 3=hinge
print("has_free =", np.any(types==mj.mjtJoint.mjJNT_FREE))
for j in range(model.njnt):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
    print(j, name, "type=", int(types[j]),
          "qpos_adr=", int(model.jnt_qposadr[j]),
          "dof_adr=",  int(model.jnt_dofadr[j]))
'''

with viewer.launch_passive(model, data) as v:
    v.sync()
    while v.is_running() and t_sim < move_time:
        pin_root(model, data)

        q_des = q_target(t_sim)        # desired pos
        q     = float(data.qpos[qadr]) # read current pos
        dq    = float(data.qvel[dadr]) # read current vel
        tau   = kp[15] * (q_des - q) - kd[15] * dq

        # 只对该自由度施加力矩
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dadr] = tau

        # 推进一步并刷新画面
        mj.mj_step(model, data)
        v.sync()

        t_sim += model.opt.timestep

        print("time = ", t_sim)
