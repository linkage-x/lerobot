#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense相机模块 - 提供对Pika设备上RealSense D405相机的访问
"""

import logging
import numpy as np

# 创建logger，但不配置全局日志系统
logger = logging.getLogger('pika.camera.realsense')

class RealSenseCamera:
    """
    RealSense相机类，提供对Pika设备上RealSense D405相机的访问
    参数:
        camera_width: 相机宽度，默认为1280
        camera_height: 相机高度，默认为720
        camera_fps: 相机帧率，默认为30
        serial_number: 相机序列号，默认为None
    """
    
    def __init__(self, camera_width=1280, camera_height=720, camera_fps=30, serial_number=None):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.serial_number = serial_number
        
        self.pipeline = None
        self.config = None
        self.is_connected = False
        
        # 尝试导入pyrealsense2库
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            logger.warning("未找到pyrealsense2库，请安装: pip install pyrealsense2")
            self.rs = None
    
    def connect(self):
        """
        连接RealSense相机
        
        返回:
            bool: 连接是否成功
        """
        if self.rs is None:
            logger.error("未找到pyrealsense2库，无法连接RealSense相机")
            return False
        
        try:    
            self.pipeline = self.rs.pipeline()
            self.config = self.rs.config()
            
            # 配置流
            self.config.enable_device(self.serial_number)
            self.config.enable_stream(self.rs.stream.depth, self.camera_width, self.camera_height, self.rs.format.z16, self.camera_fps)
            self.config.enable_stream(self.rs.stream.color, self.camera_width, self.camera_height, self.rs.format.bgr8, self.camera_fps)
            
            # 启动管道
            self.pipeline.start(self.config)
            
            self.is_connected = True
            logger.info(f"成功连接到RealSense相机，设备序列号: {self.serial_number}")
            return True
        except Exception as e:
            logger.error(f"连接RealSense相机异常: {e}")
            return False
    
    def disconnect(self):
        """
        断开RealSense相机连接
        """
        if self.pipeline and self.is_connected:
            try:
                self.pipeline.stop()
                self.is_connected = False
                logger.info(f"已断开RealSense相机连接，设备序列号: {self.serial_number}")
            except Exception as e:
                logger.error(f"断开RealSense相机连接异常: {e}")
    
    def get_frames(self):
        """
        获取一组帧（彩色和深度）
        
        返回:
            tuple: (成功标志, 彩色图像, 深度图像)
        """
        if not self.is_connected or not self.pipeline or self.rs is None:
            logger.warning("相机未连接，无法获取图像")
            return False, None, None
        
        try:
            # 等待一组连贯的帧
            frames = self.pipeline.wait_for_frames()
            
            # 获取彩色帧和深度帧
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                logger.warning("获取帧失败")
                return False, None, None
            
            # 将帧转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return True, color_image, depth_image
        except Exception as e:
            logger.error(f"获取帧异常: {e}")
            return False, None, None
    
    def get_color_frame(self):
        """
        获取彩色图像
        
        返回:
            tuple: (成功标志, 彩色图像)
        """
        success, color_image, _ = self.get_frames()
        return success, color_image
    
    def get_depth_frame(self):
        """
        获取深度图像
        
        返回:
            tuple: (成功标志, 深度图像)
        """
        success, _, depth_image = self.get_frames()
        return success, depth_image
    
    def get_camera_info(self):
        """
        获取相机信息
        
        返回:
            dict: 相机信息
        """
        if not self.is_connected or not self.pipeline or self.rs is None:
            logger.warning("相机未连接，无法获取信息")
            return {}
        
        try:
            # 获取相机内参
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            color_profile = color_frame.get_profile()
            depth_profile = depth_frame.get_profile()
            
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            return {
                'color_width': color_intrinsics.width,
                'color_height': color_intrinsics.height,
                'color_fx': color_intrinsics.fx,
                'color_fy': color_intrinsics.fy,
                'color_ppx': color_intrinsics.ppx,
                'color_ppy': color_intrinsics.ppy,
                'depth_width': depth_intrinsics.width,
                'depth_height': depth_intrinsics.height,
                'depth_fx': depth_intrinsics.fx,
                'depth_fy': depth_intrinsics.fy,
                'depth_ppx': depth_intrinsics.ppx,
                'depth_ppy': depth_intrinsics.ppy,
                'serial_number': self.serial_number
            }
        except Exception as e:
            logger.error(f"获取相机信息异常: {e}")
            return {}
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
