# -- coding: utf-8 --

import time
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


# 为ProgressThread线程定义一个函数
def progress_thread(cam=0, nMode=0):
    stFileAccessProgress = MV_CC_FILE_ACCESS_PROGRESS()
    memset(byref(stFileAccessProgress), 0, sizeof(stFileAccessProgress))
    while True:
        #ch:获取文件存取进度 |en:Get progress of file access
        ret = cam.MV_CC_GetFileAccessProgress(stFileAccessProgress)
        print ("State = [%x],Completed = [%d],Total = [%d]" % (ret, stFileAccessProgress.nCompleted, stFileAccessProgress.nTotal))
        if (ret != MV_OK or (stFileAccessProgress.nCompleted != 0 and stFileAccessProgress.nCompleted == stFileAccessProgress.nTotal)):
            print('press Enter key to continue.')
            break

# 为FileAccessThread线程定义一个函数
def file_access_thread(cam=0, nMode=0):
    stFileAccess = MV_CC_FILE_ACCESS()
    memset(byref(stFileAccess), 0, sizeof(stFileAccess))
    stFileAccess.pUserFileName = 'UserSet1.bin'.encode('ascii')
    stFileAccess.pDevFileName = 'UserSet1'.encode('ascii')
    if 1 == nMode:
        #ch:读模式 |en:Read mode
        ret = cam.MV_CC_FileAccessRead(stFileAccess)
        if MV_OK != ret:
            print ("file access read fail ret [0x%x]\n" % ret)
    elif 2 == nMode:
        #ch:写模式 |en:Write mode
        ret = cam.MV_CC_FileAccessWrite(stFileAccess)
        if MV_OK != ret:
            print ("file access write fail ret [0x%x]\n" % ret)

if __name__ == "__main__":

    try:
        # ch:初始化SDK | en: initialize SDK
        MvCamera.MV_CC_Initialize()
        
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)
        
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print ("find no Device!")
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

                strSerialNumber =  decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)                
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

        #ch:读模式 |en:Read mode
        print ("read to file.")
        print('press Enter key to start.')
        input_func()

        try:
            hReadThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 1))
            hReadThreadHandle.start()
            time.sleep(0.005)
            hProgress1ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 1))
            hProgress1ThreadHandle.start()
        except:
            raise Exception ("error: unable to start thread")

        print ("waiting.")
        input_func()

        hReadThreadHandle.join()
        hProgress1ThreadHandle.join()

        #ch:写模式 |en:Write mode
        print ("write from file.")
        print('press Enter key to start.')
        input_func()

        try:
            hWriteThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 2))
            hWriteThreadHandle.start()
            time.sleep(0.005)
            hProgress2ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 2))
            hProgress2ThreadHandle.start()
        except:
            raise Exception ("error: unable to start thread")

        print ("waiting.")
        input_func()

        hWriteThreadHandle.join()
        hProgress2ThreadHandle.join()

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception ("close deivce fail! ret[0x%x]" % ret)


        # ch:销毁句柄 | Destroy handle
        ret = cam.MV_CC_DestroyHandle()

    except Exception as e:
        print(e)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    finally:
        # ch:反初始化SDK | en: finalize SDK
        MvCamera.MV_CC_Finalize()
