#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
鱼眼相机模块 - 提供对Pika设备上鱼眼相机的访问
"""

import cv2
import logging
import numpy as np
import threading
import time
# 创建logger，但不配置全局日志系统
logger = logging.getLogger('pika.camera.fisheye')

class FisheyeCamera:
    """
    鱼眼相机类，提供对Pika设备上鱼眼相机的访问
    
    属性:
        camera_width (int): 相机宽度，默认为1280
        camera_height (int): 相机高度，默认为720
        camera_fps (int): 相机帧率，默认为30
        device_id (int): 相机设备ID，默认为0
        is_connected (bool): 设备是否连接，默认为False
    """
    
    def __init__(self, camera_width=1280, camera_height=720, camera_fps=30, device_id=0, fisheye_thread_fps=100):
        # fps use to open a high fps thread to get image
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.device_id = device_id
        self.fisheye_thread_fps = fisheye_thread_fps
        self.cap = None
        self.is_connected = False
        
        self.reading_thread = None
        self.stop_thread = False
        self.last_image_flag = False
        self.last_image = None
        self.last_image_lock = threading.Lock()
    
    def connect(self):
        """
        连接鱼眼相机
        
        返回:
            bool: 连接是否成功
        """
        try:
            cv2.setLogLevel(0)
            self.cap = cv2.VideoCapture(self.device_id)
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps) # 该设置似乎不起作用
            
            if not self.cap.isOpened():
                logger.error(f"无法打开鱼眼相机，设备ID: {self.device_id}")
                return False
            
            self.is_connected = True
            logger.info(f"成功连接到鱼眼相机，设备ID: {self.device_id}")
            self.start_reading_thread()
            
            return True
        except Exception as e:
            logger.error(f"连接鱼眼相机异常: {e}")
            return False
    
    def disconnect(self):
        """
        断开鱼眼相机连接
        """
        self.stop_reading_thread()
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"已断开鱼眼相机连接，设备ID: {self.device_id}")
    
    def start_reading_thread(self):
        """
        启动读取线程
        
        参数:
            callback (callable): 数据回调函数，接收解析后的JSON对象
        """
        if self.reading_thread and self.reading_thread.is_alive():
            logger.warning("读取线程已经在运行")
            return
        self.stop_thread = False
        self.reading_thread = threading.Thread(target=self._reading_thread_func)
        self.reading_thread.daemon = True
        self.reading_thread.start()
        
    def stop_reading_thread(self):
        """
        停止读取线程
        """
        self.stop_thread = True
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=1.0)
            logger.info("读取线程已停止")
    
    def _reading_thread_func(self):
        
        logger.info("启动鱼眼相机高频读取线程")
        time_interval = 1 / self.fisheye_thread_fps
        while not self.stop_thread:
            try:
                ret, frame = self.cap.read() # 非常高频，至少200Hz, 因此不需要计算等待
                
                if not ret:
                    logger.warning("读取图像失败")
                    break
                
                with self.last_image_lock:
                    self.last_image_flag = True
                    self.last_image = frame
                
                time.sleep(time_interval)
                
            except Exception as e:
                logger.error(f"获取图像异常: {e}")
                break
        
        logger.info("鱼眼相机高频读取线程已停止")
        
    def get_frame(self):
        return self.last_image_flag, self.last_image
    
    def get_camera_info(self):
        """
        获取相机信息
        
        返回:
            dict: 相机信息
        """
        if not self.is_connected or not self.cap:
            logger.warning("相机未连接，无法获取信息")
            return {}
        
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'device_id': self.device_id
            }
        except Exception as e:
            logger.error(f"获取相机信息异常: {e}")
            return {}
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
