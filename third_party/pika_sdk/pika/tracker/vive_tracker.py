#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vive Tracker模块 - 基于pysurvive库
提供对Vive Tracker设备位姿数据的访问接口
"""

import sys
import time
import os
import signal
import math
import threading
import queue
import logging
import numpy as np 
from .pose_utils import xyzQuaternion2matrix, xyzrpy2Mat, matrixToXYZQuaternion

# 创建logger，但不配置全局日志系统
logger = logging.getLogger('pika.tracker.vive_tracker')

# 导入pysurvive库
try:
    import pysurvive
    from pysurvive.pysurvive_generated import *
    from pysurvive import SimpleObject
except ImportError:
    logger.error("未找到pysurvive库，请确保已正确安装")
    raise ImportError("未找到pysurvive库，请确保已正确安装")

class PoseData:
    """位姿数据结构，用于存储和格式化位姿信息"""
    def __init__(self, device_name, timestamp, position, rotation):
        self.device_name = device_name
        self.timestamp = timestamp
        self.position = position  # [x, y, z]
        self.rotation = rotation  # [x, y, z, w] 四元数

    def __str__(self):
        """格式化输出位姿信息"""
        return f"{self.device_name}: T: {self.timestamp:.6f} P: {self.position[0]:9.6f}, {self.position[1]:9.6f}, {self.position[2]:9.6f} R: {self.rotation[0]:9.6f}, {self.rotation[1]:9.6f}, {self.rotation[2]:9.6f}, {self.rotation[3]:9.6f}"

class ViveTracker:
    """
    Vive Tracker设备类，提供对Vive Tracker设备位姿数据的访问接口
    
    参数:
        config_path (str, optional): 配置文件路径
        lh_config (str, optional): 灯塔配置
        args (list, optional): 其他pysurvive参数
    """
    
    def __init__(self, pose_offset=None, use_uid=False, config_path=None, lh_config=None, args=None):
        self.config_path = config_path
        self.lh_config = lh_config
        self.args = args if args else []
        
        # 初始化状态变量
        self.running = False
        self.context = None
        self.pose_queue = queue.Queue(maxsize=100)  # 用于存储最新位姿的队列
        self.devices_info = {}  # 存储设备信息的字典, 如果use uid的话key为uid否则为device name
        self.data_lock = threading.Lock()
        self.latest_poses = {}  # 存储每个设备的最新位姿
        self.use_uid = use_uid
        self._pose_offset = pose_offset
        
        # 线程对象
        self.collector_thread = None
        self.processor_thread = None
        self.device_monitor_thread = None
    
    def connect(self):
        """
        初始化并连接到Vive Tracker设备
        
        返回:
            bool: 连接是否成功
        """
        if self.running:
            logger.warning("Vive Tracker已经连接")
            return True
        
        try:
            logger.info("正在初始化pysurvive...")
            
            # 构建pysurvive参数
            survive_args = sys.argv[:1]  # 保留程序名
            
            # 添加配置文件参数
            if self.config_path:
                survive_args.extend(['--config', self.config_path])
            
            # 添加灯塔配置参数
            if self.lh_config:
                survive_args.extend(['--lh', self.lh_config])
            
            # 添加其他参数
            survive_args.extend(self.args)
            
            # 初始化pysurvive上下文
            self.context = pysurvive.SimpleContext(survive_args)
            if not self.context:
                logger.error("错误: 无法初始化pysurvive上下文")
                return False
            
            logger.info("pysurvive初始化成功")
            
            # 标记为运行状态
            self.running = True
            
            # 创建并启动位姿收集线程
            self.collector_thread = threading.Thread(target=self._pose_collector)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            
            # 创建并启动位姿处理线程
            self.processor_thread = threading.Thread(target=self._pose_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            
            # 创建并启动设备监控线程
            self.device_monitor_thread = threading.Thread(target=self._device_monitor)
            self.device_monitor_thread.daemon = True
            self.device_monitor_thread.start()
            
            logger.info("Vive Tracker位姿追踪已启动")
            
            # 等待初始数据
            time.sleep(0.5)
            return True
            
        except Exception as e:
            logger.error(f"连接Vive Tracker时发生错误: {e}")
            self.running = False
            return False
    
    def disconnect(self):
        """
        断开Vive Tracker设备连接
        """
        if not self.running:
            return
        
        logger.info("正在停止Vive Tracker位姿追踪...")
        self.running = False
        
        # 等待线程结束
        if self.collector_thread:
            self.collector_thread.join(timeout=2.0)
        
        if self.processor_thread:
            self.processor_thread.join(timeout=2.0)
            
        if self.device_monitor_thread:
            self.device_monitor_thread.join(timeout=2.0)
        
        # 清理资源
        self.context = None
        self.pose_queue = queue.Queue(maxsize=100)
        
        # 打印统计信息
        logger.info("设备统计信息:")
        for device_name, info in self.devices_info.items():
            logger.info(f"  - {device_name}: 更新次数 {info['updates']}")
        
        logger.info("Vive Tracker已断开连接")
    
    def _device_monitor(self):
        """
        设备监控线程函数
        定期检查新设备并更新设备列表
        """
        logger.info("设备监控线程已启动")
        
        # 初始化设备列表
        self._update_device_list()
        
        # 定期检查新设备
        while self.running and self.context.Running():
            # 更新设备列表
            self._update_device_list()
            
            # 每秒检查一次
            time.sleep(1.0)
    
    def _update_device_list(self):
        """
        更新设备列表
        """
        try:
            device_key_list = self._get_device_uid_list() if self.use_uid else self._get_device_name_list()
            if device_key_list is None:
                raise ValueError(f'获取device key列表出错， 是否使用uid: {self.use_uid}')
            
            # 更新设备信息字典
            with self.data_lock:
                for device_key in device_key_list:
                    if device_key not in self.devices_info:
                        logger.info(f"检测到新设备: {device_key}, 是否使用uid: {self.use_uid}")
                        self.devices_info[device_key] = {"updates": 0, "last_update": 0}
        except Exception as e:
            logger.error(f"更新设备列表时出错: {e}")

    def _get_device_name_list(self):
        """
        获取设备name
        """
        try:
            # 获取当前所有设备
            devices = list(self.context.Objects())
            
            # 获取所有设备的name list
            name_lists = []
            for device in devices:
                name = str(device.Name(), 'utf-8')
                name_lists.append(name)
            return name_lists
        except Exception as e:
            logger.error(f"更新设备name列表时出错: {e}")
            return None
    
    def _get_device_uid_list(self):
        """
        获取设备uid
        """
        try:
            # 获取当前所有设备
            devices = list(self.context.Objects())
            
            # 获取所有设备的uid list
            uid_lists = []
            for device in devices:
                cur_device_uid = str(simple_serial_number(device.ptr), 'utf-8')
                uid_lists.append(cur_device_uid)
            return uid_lists
        except Exception as e:
            logger.error(f"更新设备uid列表时出错: {e}")
            return None
        
    def _get_device_uid(self, device):
        try:
            device_uid = str(simple_serial_number(device.ptr), 'utf-8')
            return device_uid
        except Exception as e:
            logger.error(f"获取设备uid时出错: {e}")
            return None
        
    def _pose_collector(self):
        """
        位姿收集线程函数
        持续从pysurvive获取最新位姿数据并放入队列
        """
        logger.info("位姿收集线程已启动")
        
        # 获取并打印所有可用设备
        devices = list(self.context.Objects())
        if not devices:
            logger.warning("警告: 未检测到任何设备")
        else:
            logger.info(f"检测到 {len(devices)} 个设备:")
            device_key_list = self._get_device_uid_list() if self.use_uid else self._get_device_name_list()
                        
            for device_key in device_key_list:
                logger.info(f"  - {device_key} use uid: {self.use_uid}")
                self.devices_info[device_key] = {"updates": 0, "last_update": 0}
        
        # 持续获取最新位姿
        while self.running and self.context.Running():
            updated = self.context.NextUpdated()
            if updated:
                # 获取设备key
                if not self.use_uid:
                    device_name = str(updated.Name(), 'utf-8')
                else: device_name = self._get_device_uid(updated)
                
                # 如果是新设备，添加到设备信息字典
                with self.data_lock:
                    if device_name not in self.devices_info:
                        logger.info(f"检测到新设备更新: {device_name}, 是否使用uid: {self.use_uid}")
                        self.devices_info[device_name] = {"updates": 0, "last_update": 0}
                
                # 获取位姿数据
                pose_obj = updated.Pose()
                pose_data = pose_obj[0]  # 位姿数据
                timestamp = pose_obj[1]  # 时间戳
                
                # 将位姿数据转换为矩阵
                # 注意：pysurvive 里获取的四元素顺序为：[w,x,y,z]，在下面的操作中转换为[x,y,z,w]
                origin_mat = xyzQuaternion2matrix(
                                pose_data.Pos[0], pose_data.Pos[1], pose_data.Pos[2],
                                pose_data.Rot[1], pose_data.Rot[2], pose_data.Rot[3], pose_data.Rot[0]
                            )
                
                pose_offset = True
                if self._pose_offset and device_name in self._pose_offset: 
                    pose_offset = self._pose_offset[device_name]
                if pose_offset:
                    # 初始旋转校正：绕X轴旋转 -20度（roll）
                    initial_rotation = xyzrpy2Mat(0, 0, 0, -(20.0 / 180.0 * math.pi), 0, 0)
                    # 对齐旋转：绕X轴 -90度，绕Y轴 -90度
                    alignment_rotation = xyzrpy2Mat(0, 0, 0, -90/180*math.pi, -90/180*math.pi, 0)
                    # 合并旋转矩阵
                    rotate_matrix = np.dot(initial_rotation,alignment_rotation)
                    # 应用平移变换 - 将采集到的pose数据变换到夹爪中心
                    transform_matrix = xyzrpy2Mat(0.172, 0, -0.076, 0, 0, 0)
                    # 计算最终变换矩阵
                    result_mat = np.matmul(np.matmul(origin_mat, rotate_matrix), transform_matrix)
                else:
                    result_mat = origin_mat
                # 从结果矩阵中提取位置和四元数
                x, y, z, qx, qy, qz, qw = matrixToXYZQuaternion(result_mat)
                
                # 提取位置和旋转信息
                position = [x, y, z]
                rotation = [qx, qy, qz, qw]
                    
                # 创建位姿数据对象
                pose = PoseData(device_name, timestamp, position, rotation)
                
                # 更新设备信息
                with self.data_lock:
                    if device_name in self.devices_info:
                        self.devices_info[device_name]["updates"] += 1
                        self.devices_info[device_name]["last_update"] = time.time()
                
                # 将位姿数据放入队列，如果队列满则丢弃旧数据
                try:
                    self.pose_queue.put_nowait(pose)
                except queue.Full:
                    try:
                        self.pose_queue.get_nowait()  # 丢弃最旧的数据
                        self.pose_queue.put_nowait(pose)
                    except:
                        pass  # 忽略可能的错误
    
    def _pose_processor(self):
        """
        位姿处理线程函数
        从队列中获取并处理位姿数据，更新最新位姿字典
        """
        logger.info("位姿处理线程已启动")
        
        while self.running:
            try:
                # 尝试从队列获取位姿数据，设置超时以便定期检查running状态
                pose = self.pose_queue.get(timeout=0.1)
                
                # 更新最新位姿字典
                with self.data_lock:
                    self.latest_poses[pose.device_name] = pose
                
                # 在这里可以添加自定义的位姿处理逻辑
                # 例如: 发送到其他应用程序、记录到文件、进行分析等
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"处理位姿数据时出错: {e}")
    
    def get_pose(self, device_name=None):
        """
        获取指定设备的最新位姿数据
        
        参数:
            device_name (str, optional): 设备名称，如果为None则返回所有设备的位姿数据
        
        返回: 
            PoseData或dict: 如果指定了device_name，返回该设备的PoseData对象；
                          否则返回包含所有设备位姿的字典 {device_name: PoseData}
        """
        if not self.running:
            logger.warning("Vive Tracker未连接，返回空位姿数据")
            return None if device_name else {}
        
        # 强制更新一次设备列表，确保能获取到最新添加的设备
        self._update_device_list()
        
        with self.data_lock:
            if device_name:
                return self.latest_poses.get(device_name)
            else:
                return self.latest_poses.copy()
    
    def get_devices(self):
        """
        获取所有已检测到的设备列表
        
        返回:
            list: 设备名称列表
        """
        # 强制更新一次设备列表，确保能获取到最新添加的设备
        self._update_device_list()
        
        with self.data_lock:
            return list(self.devices_info.keys())
    
    def get_device_info(self, device_name=None):
        """
        获取设备信息
        
        参数:
            device_name (str, optional): 设备名称，如果为None则返回所有设备的信息
        
        返回:
            dict: 设备信息字典
        """
        # 强制更新一次设备列表，确保能获取到最新添加的设备
        self._update_device_list()
        
        with self.data_lock:
            if device_name:
                return self.devices_info.get(device_name)
            else:
                return self.devices_info.copy()
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
