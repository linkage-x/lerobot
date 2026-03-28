# -- coding: utf-8 --

import sys
import threading
import platform
import os
from ctypes import *


# 兼容不同操作系统加载 动态库
currentsystem = platform.system()
if currentsystem == 'Windows':
    sys.path.append(os.path.join(os.getenv('MVCAM_COMMON_RUNENV'), "Samples", "Python", "MvImport"))
else:
    sys.path.append(os.path.join("..", "..", "MvImport"))

from MvCameraControl_class import *


# 兼容Python 2.x和3.x的输入处理
if sys.version_info[0] < 3:
    # Python 2.x
    input_func = raw_input
else: 
    # Python 3.x
    input_func = input

# Decoding Characters
def decoding_char(ctypes_char_array):
    """
    安全地从 ctypes 字符数组中解码出字符串。
    适用于 Python 2.x 和 3.x，以及 32/64 位环境。
    """
    byte_str = memoryview(ctypes_char_array).tobytes()
    
    # 在第一个空字符处截断
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]
    
    # 多编码尝试解码
    for encoding in ['gbk', 'utf-8', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # 如果所有编码都失败，使用替换策略
    return byte_str.decode('latin-1', errors='replace')


if __name__ == "__main__":
    
    try:
        # ch:初始化SDK | en: initialize SDK
        MvCamera.MV_CC_Initialize()
        
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)
            
        interfaceList = MV_INTERFACE_INFO_LIST()
        transportLayerType = MV_GIGE_INTERFACE | MV_CAMERALINK_INTERFACE | MV_CXP_INTERFACE | MV_XOF_INTERFACE
        
        # ch:枚举采集卡 | en:Enum interfaces
        ret = MvCamera.MV_CC_EnumInterfaces(transportLayerType, interfaceList)
        if ret != 0:
            print("enum interfaces fail! ret[0x%x]" % ret)
            sys.exit()

        if interfaceList.nInterfaceNum == 0:
            print("find no interface!")
            sys.exit()

        print("Find %d interfaces!" % interfaceList.nInterfaceNum)

        for i in range(0, interfaceList.nInterfaceNum):
            interfaceInfo = cast(interfaceList.pInterfaceInfos[i], POINTER(MV_INTERFACE_INFO)).contents
            print("interface: [%d]" % i)
            
            displayName = decoding_char(interfaceInfo.chDisplayName)
            print("display name: %s" % displayName)

            serialNumber = decoding_char(interfaceInfo.chSerialNumber)
            print("serial number: %s" % serialNumber)

            modelName = decoding_char(interfaceInfo.chModelName)
            print("model name: %s" % modelName)

            interfaceId = decoding_char(interfaceInfo.chInterfaceID)
            print("interface id: %s" % interfaceId)
            
        nConnectionNum = input_func("please input the number of the interface to connect:")

        if int(nConnectionNum) >= interfaceList.nInterfaceNum:
            print("input error!")
            sys.exit()

        # ch:创建相机实例 | en:Create Camera Object
        cam = MvCamera()
        
        # ch:选择采集卡并创建句柄 | en:Select interface and create handle
        curInterface = cast(interfaceList.pInterfaceInfos[int(nConnectionNum)], POINTER(MV_INTERFACE_INFO)).contents

        ret = cam.MV_CC_CreateInterface(curInterface)
        if ret != 0:
            raise Exception("create interface handle fail! ret[0x%x]" % ret)
        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenInterface()
        if ret != 0:
            raise Exception("open interface fail! ret[0x%x]" % ret)
        else:
            print("open interface success")

        # ch:获取属性 | en:Get Feature
        stEnumValue = MVCC_ENUMVALUE()
        ret =cam.MV_CC_GetEnumValue("StreamSelector", stEnumValue)
        if ret != 0:
            raise Exception("get StreamSelector fail! ret[0x%x]" % ret)

        # ch:设置属性 | en:Set Feature
        ret = cam.MV_CC_SetEnumValue("StreamSelector", stEnumValue.nCurValue)
        if ret != 0:
            raise Exception("set StreamSelector fail! ret[0x%x]" % ret)
        else:
            print("set StreamSelector [%d] success" % stEnumValue.nCurValue)

        # ch:关闭采集卡 | en:Close interface
        ret = cam.MV_CC_CloseInterface()
        if ret != 0:
            raise Exception("close interface fail! ret[0x%x]" % ret)
        else:
            print("close interface success")

        # ch:销毁采集卡句柄 | en:Destroy interface
        cam.MV_CC_DestroyInterface()

    except Exception as e:
        print(e)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyInterface()
    finally:
        # ch:反初始化SDK | en: finalize SDK
        MvCamera.MV_CC_Finalize()
