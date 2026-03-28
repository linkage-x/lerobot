# -- coding: utf-8 --

import sys
import copy
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
        
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                      | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print ("find no device!")
            sys.exit()

        print ("find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
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

                strSerialNumber = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)                
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print ("\nCML device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print ("\nCXP device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName)
                print ("device model name: %s" % strModeName)
                
                strSerialNumber = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print ("\nXoF device: [%d]" % i)
                strModeName = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName)
                print ("device model name: %s" % strModeName)

                strSerialNumber = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber)
                print ("user serial number: %s" % strSerialNumber)

        nConnectionNum = input_func("please input the number of the device to connect:")

        if int(nConnectionNum) >= deviceList.nDeviceNum:
            print ("intput error!")
            sys.exit()

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise Exception ("create handle fail! ret[0x%x]" % ret)

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception ("open device fail! ret[0x%x]" % ret)

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise Exception ("set trigger mode fail! ret[0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception ("start grabbing fail! ret[0x%x]" % ret)

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        
        # ch:设置插值方法为均衡 | en: Set the interpolation method of Bayer format.
        ret = cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            raise Exception ("set Bayer convert quality fail! ret[0x%x]" % ret)

        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret :
            print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))

            nRGBSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stConvertParam.pSrcData = stOutFrame.pBufAddr
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType  
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nDstBufferSize = nRGBSize

            ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                raise Exception ("convert pixel fail! ret[0x%x]" % ret)

            cam.MV_CC_FreeImageBuffer(stOutFrame)

            print ("convert pixeltype succeed!")
        
            file_path = "AfterConvert_RGB.raw"   
            file_open = open(file_path.encode('ascii'), 'wb+')
            try:
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                file_open.write(img_buff)
            except:
                raise Exception("save file executed failed:%s" % e.message)
            finally:
                file_open.close() 
        else:
            print ("get one frame fail, ret[0x%x]" % ret)
            
        print ("press Enter key to stop grabbing.")
        input_func()

        # ch:停止取流 | en:Stop grab image
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception ("stop grabbing fail! ret[0x%x]" % ret)

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception ("close deivce fail! ret[0x%x]" % ret)

        # ch:销毁句柄 | Destroy handle
        cam.MV_CC_DestroyHandle()

            
    except Exception as e:
        print(e)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    finally:
        # ch:反初始化SDK | en: finalize SDK
        MvCamera.MV_CC_Finalize()
