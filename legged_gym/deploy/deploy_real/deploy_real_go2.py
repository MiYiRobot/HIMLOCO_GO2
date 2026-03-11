from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import torch.nn.functional as F
import logging

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # 加载actor和encoder
        self.policy = torch.jit.load(config.policy_path)
        # 分离出actor和encoder
        self.actor = self.policy.actor
        self.encoder = self.policy.estimator

        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        
        self.current_obs = np.zeros(config.num_obs_current, dtype=np.float32)
        self.obs_encoder = np.zeros(config.num_obs_encoder, dtype=np.float32) # 编码器网络输入
        self.obs = np.zeros(config.num_obs, dtype=np.float32) # 策略网络输入
        self.sc = SportClient()  #运动控制类
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient() #模式选择类
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()     #发送给电机的指令包
            self.low_state = unitree_go_msg_dds__LowState_() #从底层接收的状态包

            #发布者初始化
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            #订阅者初始化
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()  #阻塞等待连接

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":  #初始化机器狗发送指令
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # 关闭运控服务，否则LowCmd指令会被覆盖
    def shut_down_control_service(self):
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        print("Successfully shut down the operation control service.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_pos = self.config.default_angles
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):   #恢复到默认位置状态，等待按钮A按下
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(12):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0  # qd
                self.low_cmd.motor_cmd[motor_idx].kp = 40.0
                self.low_cmd.motor_cmd[motor_idx].kd = 0.6
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        #四元数
        quat = self.low_state.imu_state.quaternion
        #角速度
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)# imu_state quaternion: w, x, y, z  
        qj_obs = self.qj.copy() #关节位置观测
        dqj_obs = self.dqj.copy() #关节速度观测
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.current_obs[:3] = self.cmd * self.config.cmd_scale
        self.current_obs[3:6] = ang_vel
        self.current_obs[6:9] = gravity_orientation
        self.current_obs[9 : 9 + num_actions] = qj_obs
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        #np.concatenate(...)：把最新的一帧拼在最前面，把旧数据往后挤   观测历史
        self.obs_encoder = np.concatenate((self.current_obs[:config.num_obs_current], self.obs_encoder[:-config.num_obs_current]),axis=-1)
        obs_encoder_tensor = torch.from_numpy(self.obs_encoder.copy()).unsqueeze(0)#转换成tensor
        predict_tensor = self.encoder(obs_encoder_tensor)   #输入：观测历史 输出：预测的环境特性
        vel_tensor, latent_tensor = predict_tensor[..., :3], predict_tensor[..., 3:] # 拆分学生特性
        latent_tensor = F.normalize(latent_tensor, dim=-1, p=2)   #归一化处理

        #获取历史观测数据，并根据这些数据生成相应动作
        obs_tensor = torch.cat((obs_encoder_tensor[:,:config.num_obs_current], vel_tensor, latent_tensor), dim=-1)
        self.action = self.actor(obs_tensor).detach().numpy().squeeze()
        
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("net", type=str, help="network interface",default="enp4s0")
    # parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    # args = parser.parse_args()

    # Load config
    # config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/go2.yaml"
    config = Config(config_path)

    # Initialize DDS communication
    # ChannelFactoryInitialize(0, args.net)   #初始化SDK通讯通道
    ChannelFactoryInitialize(0, "enp4s0")   #初始化SDK通讯通道

    controller = Controller(config)   #创建控制器

    controller.shut_down_control_service() #关闭运控模式

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    #Wait for Singal A to start
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:  # 捕获 Ctrl+C，准备安全退出
            print("收到中断信号，正在安全停止机器狗...")
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
#cd HIMLOCO_GO2/legged_gym/deploy/deploy_real
#pythom deploy_real_go2.py enp4s0 
#启动顺序，先按Start站立，再按A启动，按select停止
