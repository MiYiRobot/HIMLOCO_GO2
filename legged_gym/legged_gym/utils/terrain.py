# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        #cfg.terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        #self.proportions = [0.1, 0.3, 0.6, 0.9, 1.0]
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # num_sub_terrains:总网格数量:整个地形被分成 num_rows × num_cols 个小区域。
        # p.s.实例变量（self.xxx）可以在运行时动态添加，
        # 即使 cfg（也就是 LeggedRobotCfg.terrain）在定义时没有 num_sub_terrains，也可以在实例上动态创建这个属性。
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # env_origins:存储每个地形区域的起点坐标，初始化为全零。
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 每一个小网格长/宽方向像素数(每一个像素坐标对应某处地形某一个点)=总宽度/地形的水平分辨率(1 像素代表多少米)
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)  #8/0.1
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale) #8/0.1
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)    #25/0.1

        #row(x/length)方向在相邻子地形之间留出 0.5m 平台间隔（不截断子地形内容）
        self.row_gap_size_m = self.cfg.row_gap_size_m 
        self.row_gap_px = int(self.row_gap_size_m / cfg.horizontal_scale)
        # 子地形在 row 方向的放置步长（像素）
        self.row_stride_px = self.length_per_env_pixels + 2 * self.row_gap_px
        # 同样对应的“米”单位步长，用于 env_origin_x
        self.row_stride_m = self.env_length + 2.0 * self.row_gap_size_m
        # ==========================================================================================


        # 地形行方向总像素=地形行数*每行像素数+2*边界像素
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border #(20*80)+2*250
        # self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border# (10*80)+2*250
        self.tot_rows = int(cfg.num_rows * self.row_stride_px) + 2 * self.border

        # 我们给每一个像素一个高度数据，所以先建立一个tot_rows行，tot_cols列的矩阵
        # 初始值全零，这个就是原始高度场，很重要，不要忘记这个变量！！！
        # 另外，这是个整形，就是因为byd isaacgym官方文档输入地形参数的接口必须给整形！！！
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        # 如果地形类型是 "trimesh"，则调用 convert_heightfield_to_trimesh 函数
        # 将高度场转换为三角网格（trimesh），并存储相应的顶点和三角形数据。
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 根据行列给出不同的难度和选择值，行列越大，难度越大，选择值越大
                difficulty = i / self.cfg.num_rows  # i/10
                choice = j / self.cfg.num_cols + 0.001  # j/20 + 0.001

                # 有难度和选择值给每一个具体的网格生成地形
                terrain = self.make_terrain(choice, difficulty)
                 # 把生成的地形添加到地图中。
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        '''
          给每一个具体的网格生成地形。
          make_terrain 方法根据传入的 choice 和 difficulty 生成地形。
          该方法首先创建一个 SubTerrain 对象，然后根据 choice 和 difficulty 生成不同类型的地形。
        '''

        # p.s.这里为什么要创建一个SubTerrain对象呢？因为要用isaacgym官方封装的库函数(terrain_utils.py)
        # 来生成地形。所以想调用这些函数就必须先创建一个 SubTerrain 对象。
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4    #倾斜度，随着难度增加，倾斜度逐渐加大。
        amplitude = 0.01 + 0.07 * difficulty  #噪声
        
        step_height = 0.06 + 0.15 * difficulty  #步高，随着难度增加，步高增加
        # step_height = 0.05 + 0.18 * difficulty  #步高，随着难度增加，步高增加

        discrete_obstacles_height = 0.05 + difficulty * 0.1  #障碍物的高度
        stepping_stones_size = 1.5 * (1.05 - difficulty) #跳石的大小，随着难度增加，跳石的大小减小
        stone_distance = 0.05 if difficulty==0 else 0.1  #跳石间距，难度为 0 时较小，其他难度较大
        gap_size = 1. * difficulty  #间隙大小，随着难度增加而增大
        pit_depth = 1. * difficulty #坑深，随着难度增加，坑深增加

        # choice 用来决定生成哪种类型的地形。它的值与 self.proportions 的数组进行比较，
        # 从而选择不同的地形生成方法。每个 choice 范围对应一种特定的地形类型。
        #斜坡
        if choice < self.proportions[0]:
                # choice < self.proportions[0] / 2 则反转斜坡的方向（通过 slope *= -1）下坡
            # self.proportions[0]/2 < choice < self.proportions[0] 上坡
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #崎岖斜坡
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # 增加随机噪声
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        # 上下樓梯
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
        # 生成离散地形（石板地形）
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        # elif choice < self.proportions[5]:
        #     terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        # elif choice < self.proportions[6]:
        #     gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        # else:
        #     pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        # start_x, end_x：通过 i（行号）计算地形在 x 方向的起始和结束位置，单位是像素。
        # self.length_per_env_pixels 代表每个网格的宽度，self.border 是地图的边界偏移量
        start_x = self.border + i * self.row_stride_px + self.row_gap_px
        end_x = start_x + self.length_per_env_pixels
        # start_x = self.border + i * self.length_per_env_pixels
        # end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        # 将生成的地形高度数据添加到地图
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x 和 env_origin_y：计算环境的原点位置（中心点），
        # 分别基于行 i 和列 j 的位置。self.env_length 和 self.env_width 分别表示每个环境的长度和宽度。
        env_origin_x = (i + 0.5) * self.row_stride_m
        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # x1, x2, y1, y2：这些是通过环境的大小、水平缩放比例来确定的索引，
        # 表示环境中心区域的范围。这些索引用于在 terrain.height_field_raw 中查找环境的最大高度。
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # 通过在 terrain.height_field_raw 中查找环境中心区域的最大高度来计算该环境的 z 坐标（高度），
        # 然后将其乘以 vertical_scale 来进行缩放。
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        # 最后，将计算出的环境原点的坐标 [env_origin_x, env_origin_y, env_origin_z] 
        # 存储到 self.env_origins 数组中，表示该地形在整个地图上的位置。 env_origins记录的是机器人初始刷新的位置坐标
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
