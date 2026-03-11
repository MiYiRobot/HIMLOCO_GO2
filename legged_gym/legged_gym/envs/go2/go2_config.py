from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class env(LeggedRobotCfg.env ):
        num_envs = 2048

    class domain_rand(LeggedRobotCfg.domain_rand):
        #躯干质量随机化
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]
        # 随机化质心位置
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        #大小腿质量随机化
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]
        #地面摩擦系数随机化
        randomize_friction = True
        # friction_range = [0.3, 1.25]
        friction_range = [0.03, 1.25]
        #地面弹性随机化
        randomize_restitution = True
        restitution_range = [0., 1.0]
         #关节特性随机化
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        #pd参数随机化
        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]
        #随机化初始关节位置
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        #持续外力
        disturbance = True
        # disturbance_range = [-8.0, 8.0]
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        #随机推机器人
        push_robots = True
        push_interval_s = 16
        # max_push_vel_xy = 0.5
        max_push_vel_xy = 1.

        delay = True

    class terrain( LeggedRobotCfg.terrain ):
        num_rows = 20   # 原来是 10，控制地形难度递进的级数
        num_cols = 10   # 原来是 20，控制地形种类的宽度
        # num_rows = 10   # 原来是 10，控制地形难度递进的级数
        # num_cols = 20   # 原来是 20，控制地形种类的宽度
        max_init_terrain_level = 5 # starting curriculum state 原来是5
        row_gap_size_m = 0.5
        # 总地形块就从 200块 (10x20) 变成了 50块 (5x10)

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class commands( LeggedRobotCfg.commands ):
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.5, 1.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        #速度追踪函数的指数函数参数，sigma 越小：误差稍大奖励就迅速趋近 0 sigma越大则允许一定误差仍有奖励
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        #关节位置、速度和力矩活动软范围
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

        base_height_target = 0.33  #基座高度
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.22   #脚相对于基座抬起的高度
        # feet_air_time =  1.0
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            #线速度、角速度追踪
            tracking_lin_vel = 2.5
            # tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # tracking_ang_vel = 0.5

            lin_vel_z_up = -2.0   #惩罚在z方向上的线速度
            # lin_vel_z = -2.0
            ang_vel_xy_up = -0.1  #惩罚 roll/pitch 方向的角速度
            orientation_up = -0.2 #惩罚机体不水平
            dof_acc = -2.5e-7  #惩罚关节加速度
            joint_power = -2e-5   #惩罚功率/能耗风格
            base_height = -4.0   #惩罚 base 高度偏离 target
            foot_clearance_up = -0.2  #抬腿高度惩罚，如果摆动的脚没有达到相应的高度就扣分
            action_rate = -0.01  #惩罚动作变化过快
            smoothness = -0.01   #二阶差分平滑（抑制加加速度/急变）
            # feet_air_time =  0.0
            feet_air_time =  1.25  #奖励抬腿
            collision_up = -1.0  #惩罚碰撞
            feet_stumble = -0.0  #没这个函数
            stand_still = -0.15  #命令接近于0时希望静止时姿态接近默认姿态
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            feet_contact_forces = -0.00015 #接触力过大惩罚
            foot_slide_up = -0.01  #打滑惩罚，脚接触地面时，其线速度应该为0

    class noise( LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            height_measurements = 0.05

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3 #5.e-4  

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        max_iterations = 2000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations


  
