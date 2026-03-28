# -- coding: utf-8 --

import sys
import os
import platform

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
    
fun_ctype = get_platform_functype()
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = fun_ctype(None, pData, stFrameInfo, c_void_p)

def image_callback(pData, pFrameInfo, pUser):
        stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
        if stFrameInfo:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))

CALL_BACK_FUN = FrameInfoCallBack(image_callback)


def check_feature_node_access(cam_ins, node_name):
    access_mode = MV_XML_AccessMode()
    res = cam_ins.MV_XML_GetNodeAccessMode(node_name, access_mode)
    if res != 0:
        return False
    if access_mode == AM_WO or access_mode == AM_RO or access_mode == AM_RW:
        return True
    else:
        return False


if __name__ == "__main__":

    try:
        # ch:初始化SDK | en: initialize SDK
        MvCamera.MV_CC_Initialize()

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE)

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                print ("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber =  decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)                
                print ("user serial number: %s" % strSerialNumber)


        nConnectionNum = input_func("please input the number of the device to connect:")

        if int(nConnectionNum) >= deviceList.nDeviceNum:
            print("input error!")
            sys.exit()

        nTriggerIndex = input_func("Please Input trigger selector index: 0-FrameTrigger, 1-LineTrigger:")
        if int(nTriggerIndex) not in {0, 1}:
            print("input error!")
            sys.exit()

        # ch:创建相机实例 | en:Create Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise Exception("create handle fail! ret[0x%x]" % ret)

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception("open device fail! ret[0x%x]" % ret)

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:线阵相机帧触发预设置 | en:Settings in FrameScan mode
        if int(nTriggerIndex) == 0:
            ret = cam.MV_CC_SetEnumValueByString("ScanMode", "FrameScan")
            if ret == 0:
                print("set frame scan mode")

            # ch: 判断FrameTriggerControl是否可读 | en: Check if FrameTriggerControl is readable
            if check_feature_node_access(cam, "FrameTriggerControl"):
                # ch:设置触发模式为on | en:Set trigger mode as on
                ret = cam.MV_CC_SetBoolValue("FrameTriggerMode", True)
                if ret != 0:
                    raise Exception("set frame trigger mode on fail! ret[0x%x]" % ret)
                # ch:设置触发源为Line0 | en:Set trigger source as Line0
                ret = cam.MV_CC_SetEnumValue("FrameTriggerSource", 0)
                if ret != 0:
                    raise Exception("set trigger source line0 fail! ret[0x%x]" % ret)
            else:
                # ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
                ret = cam.MV_CC_SetEnumValue("TriggerSelector", 6)
                if ret != 0:
                    raise Exception("set trigger selector fail! ret[0x%x]" % ret)

                # ch:设置触发模式为on | en:Set trigger mode as on
                ret = cam.MV_CC_SetEnumValue("TriggerMode", 1)
                if ret != 0:
                    raise Exception("set trigger mode fail! ret[0x%x]" % ret)

                # ch:设置触发源为Line0 | en:Set trigger source as Line0
                ret = cam.MV_CC_SetEnumValue("TriggerSource", 0)
                if ret != 0:
                    raise Exception("set trigger source fail! ret[0x%x]" % ret)
        # ch:线阵相机行触发预设置 | en:Settings in LineScan mode
        else:
            # ch: 判断FrameTriggerControl是否可读 | en: Check if FrameTriggerControl is readable
            if check_feature_node_access(cam, "LineTriggerControl"):
                # ch:设置触发模式为on | en:Set trigger mode as on
                ret = cam.MV_CC_SetBoolValue("LineTriggerMode", True)
                if ret != 0:
                    raise Exception("set trigger mode fail! ret[0x%x]" % ret)
                # ch:设置触发源为EncoderModuleOut | en:Set trigger source as EncoderModuleOut
                ret = cam.MV_CC_SetEnumValue("LineTriggerSource", 6)
                if ret != 0:
                    raise Exception("set trigger source fail! ret[0x%x]" % ret)
            else:
                # ch:设置触发选项为LineStart | en:Set trigger selector as LineStart
                ret = cam.MV_CC_SetEnumValue("TriggerSelector", 9)
                if ret != 0:
                    raise Exception("set trigger selector fail! ret[0x%x]" % ret)

                # ch:设置触发模式为on | en:Set trigger mode as on
                ret = cam.MV_CC_SetEnumValue("TriggerMode", 1)
                if ret != 0:
                    raise Exception("set trigger mode fail! ret[0x%x]" % ret)

                # ch:设置触发源为EncoderModuleOut | en:Set trigger source as EncoderModuleOut
                ret = cam.MV_CC_SetEnumValue("TriggerSource", 6)
                if ret != 0:
                    raise Exception("set trigger source fail! ret[0x%x]" % ret)

            # ch:设置编码器选项为Encoder0 | en:Set encoder selector as Encoder0
            ret = cam.MV_CC_SetEnumValue("EncoderSelector", 0)
            if ret != 0:
                raise Exception("set encoder selector fail! ret[0x%x]" % ret)

            # ch:设置编码器数据源A为Line1 | en:Set encoder source A as Line1
            ret = cam.MV_CC_SetEnumValue("EncoderSourceA", 1)
            if ret != 0:
                raise Exception("set encoder sourceA fail! ret[0x%x]" % ret)

            # ch:设置编码器数据源B为Line3 | en:Set encoder source B as Line3
            ret = cam.MV_CC_SetEnumValue("EncoderSourceB", 3)
            if ret != 0:
                raise Exception("set encoder sourceB fail! ret[0x%x]" % ret)

        # ch:注册抓图回调 | en:Register image callback
        ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
        if ret != 0:
            raise Exception("register image callback fail! ret[0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("start grabbing fail! ret[0x%x]" % ret)

        print ("press Enter key to stop grabbing.")
        input_func()

        # ch:停止取流 | en:Stop grab image
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception("stop grabbing fail! ret[0x%x]" % ret)

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception("close device fail! ret[0x%x]" % ret)

        # ch:销毁句柄 | Destroy handle
        cam.MV_CC_DestroyHandle()

    except Exception as e:
        print(e)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    finally:
        # ch:反初始化SDK | en: finalize SDK
        MvCamera.MV_CC_Finalize()
