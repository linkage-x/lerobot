# -- coding: utf-8 --

import sys
import platform
import threading
import os
from ctypes import *
import numpy
import cv2

currentsystem = platform.system()
if currentsystem == 'Windows':
    sys.path.append(os.getenv('MVCAM_COMMON_RUNENV') + "/Samples/Python/MvImport")
else:
    # Demo 目录:  "/opt/xxx/Samples/平台/OpenCV/ZZ/ ")   python接口目录： "/opt/xxx/Samples/平台/Python/MvImport"
    sys.path.append("./../../../Python/MvImport")


from MvCameraControl_class import *
g_bExit = False

# 兼容Python 2.x和3.x的输入处理
if sys.version_info[0] < 3: 
    # Python 2.x
    input_func = raw_input
else: 
    # Python 3.x
    input_func = input


def IsHBPixelFormat(enPixelType = 0):
    if enPixelType in (PixelType_Gvsp_HB_Mono8, \
                        PixelType_Gvsp_HB_Mono10,\
                        PixelType_Gvsp_HB_Mono10_Packed,\
                        PixelType_Gvsp_HB_Mono12,\
                        PixelType_Gvsp_HB_Mono12_Packed,\
                        PixelType_Gvsp_HB_Mono16,\
                        PixelType_Gvsp_HB_RGB8_Packed,\
                        PixelType_Gvsp_HB_BGR8_Packed,\
                        PixelType_Gvsp_HB_RGBA8_Packed,\
                        PixelType_Gvsp_HB_BGRA8_Packed,\
                        PixelType_Gvsp_HB_RGB16_Packed,\
                        PixelType_Gvsp_HB_BGR16_Packed,\
                        PixelType_Gvsp_HB_RGBA16_Packed,\
                        PixelType_Gvsp_HB_BGRA16_Packed,\
                        PixelType_Gvsp_HB_YUV422_Packed,\
                        PixelType_Gvsp_HB_YUV422_YUYV_Packed,\
                        PixelType_Gvsp_HB_BayerGR8,\
                        PixelType_Gvsp_HB_BayerRG8,\
                        PixelType_Gvsp_HB_BayerGB8,\
                        PixelType_Gvsp_HB_BayerBG8,\
                        PixelType_Gvsp_HB_BayerRBGG8,\
                        PixelType_Gvsp_HB_BayerGB10,\
                        PixelType_Gvsp_HB_BayerGB10_Packed,\
                        PixelType_Gvsp_HB_BayerBG10,\
                        PixelType_Gvsp_HB_BayerBG10_Packed,\
                        PixelType_Gvsp_HB_BayerRG10,\
                        PixelType_Gvsp_HB_BayerRG10_Packed,\
                        PixelType_Gvsp_HB_BayerGR10,\
                        PixelType_Gvsp_HB_BayerGR10_Packed,\
                        PixelType_Gvsp_HB_BayerGB12,\
                        PixelType_Gvsp_HB_BayerGB12_Packed,\
                        PixelType_Gvsp_HB_BayerBG12,\
                        PixelType_Gvsp_HB_BayerBG12_Packed,\
                        PixelType_Gvsp_HB_BayerRG12,\
                        PixelType_Gvsp_HB_BayerRG12_Packed,\
                        PixelType_Gvsp_HB_BayerGR12,\
                        PixelType_Gvsp_HB_BayerGR12_Packed):
        return True
    else:
        return False
    

def IsMonoPixelFormat(enPixelType = 0):
    if enPixelType in (PixelType_Gvsp_Mono8, \
                        PixelType_Gvsp_Mono10, \
                        PixelType_Gvsp_Mono10_Packed, \
                        PixelType_Gvsp_Mono12, \
                        PixelType_Gvsp_Mono12_Packed, \
                        PixelType_Gvsp_Mono14, \
                        PixelType_Gvsp_Mono16):
        return True
    else:
        return False

# 为线程定义一个函数
def work_thread(cam=0):
    stOutFrame = MV_FRAME_OUT()  
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
     
    
    #执行1次
    SaveImageOnce = False
    while True:
    
        if g_bExit == True:
            break
            
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print ("get one frame: Width[%d], Height[%d], nFrameNum[%d], nLostPacket[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum, stOutFrame.stFrameInfo.nLostPacket))
            
            stDecodeParam = MV_CC_HB_DECODE_PARAM()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            
            # 如果是HB图像，需要进行HB解码
            result = IsHBPixelFormat(stOutFrame.stFrameInfo.enPixelType)
            if True == result :
                DecodeNeedBufferlen = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
                DecodeBuffer = (c_ubyte * DecodeNeedBufferlen)()
                
                stDecodeParam.pSrcBuf = stOutFrame.pBufAddr
                stDecodeParam.nSrcLen = stOutFrame.stFrameInfo.nFrameLen
                stDecodeParam.pDstBuf = DecodeBuffer
                stDecodeParam.nDstBufSize = DecodeNeedBufferlen
                ret = cam.MV_CC_HBDecode(stDecodeParam)
                if ret != 0:
                    print("HB Decode fail! ret[0x%x]" % ret)
                    cam.MV_CC_FreeImageBuffer(stOutFrame)
                    continue

                stConvertParam.pSrcData = stDecodeParam.pDstBuf
                stConvertParam.nSrcDataLen = stDecodeParam.nDstBufLen
                stConvertParam.enSrcPixelType = stDecodeParam.enDstPixelType  
            else:
                stConvertParam.pSrcData = stOutFrame.pBufAddr
                stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType  
           
            
            # 图像进行格式转换 
            enDstPixelType = PixelType_Gvsp_Undefined
            nChannelNum = 1
            bMono = IsMonoPixelFormat(stConvertParam.enSrcPixelType)
            if True == bMono :
                enDstPixelType = PixelType_Gvsp_Mono8
                nChannelNum = 1
            else:
                enDstPixelType = PixelType_Gvsp_RGB8_Packed
                nChannelNum = 3
            
            convertdestBufflen = nChannelNum *stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight 
            DstBuffer = (c_ubyte * convertdestBufflen)()
            
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight

            stConvertParam.enDstPixelType = enDstPixelType
            stConvertParam.pDstBuffer = DstBuffer
            stConvertParam.nDstBufferSize = convertdestBufflen

            ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                print ("convert pixel fail! ret[0x%x]" % ret)
                cam.MV_CC_FreeImageBuffer(stOutFrame)
                continue
        
            #转换 numpy 格式  
            if nChannelNum == 1:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=convertdestBufflen). \
                    reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth)
                
                # 执行1次, 只保存1张图
                if SaveImageOnce == False:
                    output_path = "output_mono_image.bmp"
                    cv2.imwrite(output_path, numpy_image)  
                    print("save output_mono_image.bmp success")
                    
                    SaveImageOnce = True
            else:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=convertdestBufflen). \
                    reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, 3)
                
                # 执行1次, 只保存1张图
                if SaveImageOnce == False:
                    output_path = "output_color_image.bmp"
                    cv2.imwrite(output_path, numpy_image)  
                    print("save output_color_image.bmp success")
       
                    SaveImageOnce = True

            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            print ("no data[0x%x]" % ret)
            


if __name__ == "__main__":

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    # ch:选择枚举的传输层协议  (根据需求和系统支持情况选择)    | en: Select the Transport Layer Protocol (based on your requirements and system support).
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

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0])                
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            print ("\nCML device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            print ("\nCXP device: [%d]" % i)
            strModeName =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)
            
            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
            print ("\nXoF device: [%d]" % i)
            strModeName =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = input_func("please input_func the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    
    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    
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
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam,))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")
        
    print ("press Enter key to stop grabbing.")
    input_func()

    g_bExit = True
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()

