# -- coding: utf-8 --

import sys
import threading
import platform
import os

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
    

g_bExit = False
# 为线程定义一个函数
def work_thread(cam=0, pData=0, nDataSize=0):
    out_frame_info = MV_FRAME_OUT()
    memset(byref(out_frame_info), 0, sizeof(out_frame_info))
    while True:
        ret = cam.MV_CC_GetImageBuffer(out_frame_info, 1000)
        if out_frame_info.pBufAddr is not None and 0 == ret:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                out_frame_info.stFrameInfo.nWidth, out_frame_info.stFrameInfo.nHeight,
                out_frame_info.stFrameInfo.nFrameNum))
            ret = cam.MV_CC_FreeImageBuffer(out_frame_info)
        else:
            print("error: no data[0x%x]" % ret)
        if g_bExit is True:
            break


if __name__ == "__main__":

    try:
        # ch:初始化SDK | en: initialize SDK
        MvCamera.MV_CC_Initialize()

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)
        
        device_list = MV_CC_DEVICE_INFO_LIST()
        t_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                        | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(t_layer_type, device_list)
        if ret != 0:
            print("error: enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if device_list.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % device_list.nDeviceNum)

        for i in range(0, device_list.nDeviceNum):
            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                print ("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber =  decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)                
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print ("\nCML device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print ("\nCXP device: [%d]" % i)
                strModeName =  decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName)
                print ("device model name: %s" % strModeName)
                
                strSerialNumber =  decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print ("\nXoF device: [%d]" % i)
                strModeName =  decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber =  decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)

        connection_number = input_func("please input the number of the device to connect:")

        if int(connection_number) >= device_list.nDeviceNum:
            print("error: input error!")
            sys.exit()

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        current_device_info = cast(device_list.pDeviceInfo[int(connection_number)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(current_device_info)
        if ret != 0:
            raise Exception("error: create handle fail! ret[0x%x]" % ret)

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception("error: open device fail! ret[0x%x]" % ret)

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if current_device_info.nTLayerType == MV_GIGE_DEVICE or current_device_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        b_enable = c_bool(False)
        ret = cam.MV_CC_GetBoolValue("AcquisitionLineRateEnable", b_enable)
        if ret != 0:
            raise Exception("error: get AcquisitionLineRateEnable fail! ret[0x%x]" % ret)

        # ch:设置行触发 | en:Set trigger selector is line start
        ret = cam.MV_CC_SetEnumValueByString("TriggerSelector", "LineStart")
        if ret != 0:
            raise Exception("error: set trigger selector fail! ret[0x%x]" % ret)

        # ch:设置触发模式为On | en:Set trigger mode as on
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            raise Exception("error: set trigger mode fail! ret[0x%x]" % ret)

        # ch:设置触发源 | en:Set trigger source
        ret = cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE1)
        if ret != 0:
            raise Exception("error: set trigger source fail! ret[0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("error: start grabbing fail! ret[0x%x]" % ret)

        try:
            thread_handle = threading.Thread(target=work_thread, args=(cam, None, None))
            thread_handle.start()
        except:
            raise Exception("error: unable to start thread")
            
        print ("press Enter key to stop grabbing.")
        input_func()

        g_bExit = True
        thread_handle.join()

        # ch:停止取流 | en:Stop grab image
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception("error: stop grabbing fail! ret[0x%x]" % ret)

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception("error: close device fail! ret[0x%x]" % ret)

        # ch:销毁句柄 | Destroy handle
        ret = cam.MV_CC_DestroyHandle()

    except Exception as e:
        print(e)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    finally:
        # ch:反初始化SDK | en: finalize SDK
        MvCamera.MV_CC_Finalize()
