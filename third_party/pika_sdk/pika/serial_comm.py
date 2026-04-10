#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
串口通信模块，用于与Pika系列设备进行通信
"""

import threading
import time
import json
import serial
import logging
import re # 导入re模块用于正则表达式
import struct # 导入struct模块

# 创建logger，但不配置全局日志系统
logger = logging.getLogger("pika.serial_comm")

class SerialComm:
    """
    串口通信类，负责与设备的串口通信
    
    参数:
        port (str): 串口设备路径，默认为'/dev/ttyUSB0'
        baudrate (int): 波特率，默认为460800
        timeout (float): 超时时间，默认为1.0秒
    """
    def __init__(self, port=r"/dev/ttyUSB0", baudrate=460800, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.is_connected = False
        self.reading_thread = None
        self.stop_thread = False
        self.buffer = ""
        self.callback = None
        self.data_lock = threading.Lock()
        self.latest_data = {}
        # Prevent unbounded buffer growth under noisy serial lines.
        self._max_buffer_len = 4096
    
    def connect(self):
        """
        连接串口设备
        
        返回:
            bool: 连接是否成功
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.is_connected = True
            logger.info(f"成功连接到串口设备: {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"连接串口设备失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """
        断开串口连接
        """
        self.stop_reading_thread()
        if self.serial and self.is_connected:
            self.serial.close()
            self.is_connected = False
            logger.info(f"已断开串口设备连接: {self.port}")
    
    def send_data(self, data):
        """
        发送数据到串口
        
        参数:
            data (bytes): 要发送的数据
            
        返回:
            bool: 发送是否成功
        """
        if not self.is_connected or not self.serial:
            logger.error("串口未连接，无法发送数据")
            return False
        
        try:
            self.serial.write(data)
            self.serial.flush()
            return True
        except serial.SerialException as e:
            logger.error(f"发送数据失败: {e}")
            return False
    
    def send_command(self, command_type, value=0, big_endian=False):
        """
        发送命令到设备
        
        参数:
            command_type (int): 命令类型
            value (float): 命令值，默认为0.0
            big_endian (bool): 是否使用大端序，默认为False（小端序）
            
        返回:
            bool: 发送是否成功
        """ 
        try:
            # 构建命令数据
            data = bytearray()
            data.append(command_type)  # 命令类型
            
            if big_endian:
                value_bytes = bytearray(struct.pack('>i', value))  # 大端序
            else:
                value_bytes = bytearray(struct.pack('<f', value))  # 小端序
            
            data.extend(value_bytes)
            
            # 添加结束符 \r\n
            data.extend(b'\r\n')
            
            return self.send_data(data)
        except Exception as e:
            logger.error(f"构建命令数据失败: {e}")
            return False
        
    def get_device_info_command(self):
        """
        下发GET_INFO\r\n命令到设备
        
        返回:
            bool: 发送是否成功
        """
        try:
            # 构建GET_INFO命令数据
            command = 'GET_INFO\r\n'
            data = command.encode('utf-8')
            
            return self.send_data(data)
        except Exception as e:
            logger.error(f"发送GET_INFO命令失败: {e}")
            return False
        
    def read_data(self):
        """
        从串口读取数据
        
        返回:
            bytes: 读取到的数据
        """
        if not self.is_connected or not self.serial:
            logger.error("串口未连接，无法读取数据")
            return b''
        
        try:
            if self.serial.in_waiting > 0:
                return self.serial.read(self.serial.in_waiting)
            return b''
        except serial.SerialException as e:
            logger.error(f"读取数据失败: {e}")
            return b''
    
    def _reading_thread_func(self):
        """
        读取线程函数，持续从串口读取数据并解析
        """
        logger.info("启动串口读取线程")
        while not self.stop_thread:
            if not self.is_connected:
                time.sleep(0.1)
                continue
            
            try:
                # 读取数据
                data = self.read_data()
                if data:
                    # 将读取到的数据添加到缓冲区
                    self.buffer += data.decode('utf-8', errors='ignore')
                    
                    # 查找完整的JSON对象
                    json_data = self._find_json()
                    if json_data:
                        # 如果设置了回调函数，则调用回调函数
                        if self.callback:
                            self.callback(json_data)
                        
                        # 更新最新数据
                        with self.data_lock:
                            self.latest_data = json_data
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"读取线程异常: {e}")
                time.sleep(0.1)
        
        logger.info("串口读取线程已停止")
    
    def _find_json(self):
        """
        在缓冲区中查找完整的JSON对象
        
        返回:
            dict: 解析到的JSON对象，如果没有找到则返回None
        """
        try:
            # 查找JSON对象的开始位置
            buf = self.buffer
            start = buf.find('{')
            if start == -1:
                if len(buf) > self._max_buffer_len:
                    self.buffer = ""
                return None
            if start > 0:
                # 丢弃噪声前缀，仅保留可能的JSON起点
                buf = buf[start:]
                start = 0
            
            # 使用括号计数并跳过字符串内容，避免字符串内的花括号干扰
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(buf)):
                ch = buf[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == '\\':
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                else:
                    if ch == '"':
                        in_string = True
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = buf[start:i+1]
                            remainder = buf[i+1:]

                            # --- 关键修改：处理多余的逗号 ---
                            cleaned_json_str = re.sub(r',\s*}', '}', json_str)
                            cleaned_json_str = re.sub(r',\s*\]', ']', cleaned_json_str)

                            parsed = json.loads(cleaned_json_str)
                            self.buffer = remainder
                            return parsed

            # 未找到完整JSON对象，保留缓冲区（尽量保留起始位置后的内容）
            if len(buf) - start > self._max_buffer_len:
                self.buffer = buf[start:][-self._max_buffer_len:]
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            # 丢弃当前起始符，尝试从后续数据中恢复
            self.buffer = buf[start+1:] if start >= 0 else ""
            return None
        except Exception as e:
            logger.error(f"通信Json异常: {e}")
            self.buffer = ""
            return None
    
    def start_reading_thread(self, callback=None):
        """
        启动读取线程
        
        参数:
            callback (callable): 数据回调函数，接收解析后的JSON对象
        """
        if self.reading_thread and self.reading_thread.is_alive():
            logger.warning("读取线程已经在运行")
            return
        
        self.callback = callback
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
    
    def get_latest_data(self):
        """
        获取最新的数据
        
        返回:
            dict: 最新的数据
        """
        with self.data_lock:
            return self.latest_data.copy()
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
