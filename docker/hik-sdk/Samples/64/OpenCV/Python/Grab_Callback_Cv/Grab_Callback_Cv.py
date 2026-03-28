# -- coding: utf-8 --

import sys
import platform
import copy
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

# 兼容Python 2.x和3.x的输入处理
if sys.version_info[0] < 3: 
    # Python 2.x
    input_func = raw_input
else: 
    # Python 3.x
    input_func = input
    
stFrame = POINTER(MV_FRAME_OUT)
fun_ctype = get_platform_functype()
FrameInfoCallBack2 = fun_ctype(None, stFrame, c_void_p, c_bool)


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


#执行1次
g_SaveImageOnce = False
def image_callback2(pstFrame, pUser, bAutoFree):
    stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
    User = ctypes.cast(pUser, ctypes.py_object).value
    if stFrame:
        print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]  len[%d]  nLostPacket[%d]" % (stFrame.stFrameInfo.nWidth, stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nFrameNum ,stFrame.stFrameInfo.nFrameLen, stFrame.stFrameInfo.nLostPacket))
        global g_SaveImageOnce

        while True:
       
            stDecodeParam = MV_CC_HB_DECODE_PARAM()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            
            # 如果是HB图像，需要进行HB解码
            result = IsHBPixelFormat(stFrame.stFrameInfo.enPixelType)
            if True == result :
                DecodeBufferlen = stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight * 3
                DecodeBuffer = (c_ubyte * DecodeBufferlen)()

                
                stDecodeParam.pSrcBuf = stFrame.pBufAddr
                stDecodeParam.nSrcLen = stFrame.stFrameInfo.nFrameLen
                stDecodeParam.pDstBuf = DecodeBuffer
                stDecodeParam.nDstBufSize = DecodeBufferlen
                ret = User.MV_CC_HBDecode(stDecodeParam)
                if ret != 0:
                    print("HB Decode fail! ret[0x%x]" % ret)
                    break
                    
                stConvertParam.pSrcData = stDecodeParam.pDstBuf
                stConvertParam.nSrcDataLen = stDecodeParam.nDstBufLen
                stConvertParam.enSrcPixelType = stDecodeParam.enDstPixelType  
            else:
                stConvertParam.pSrcData = stFrame.pBufAddr
                stConvertParam.nSrcDataLen = stFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrame.stFrameInfo.enPixelType  
           
            
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
            
            convertdestBufflen = nChannelNum *stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight 
            
            DstBuffer = (c_ubyte * convertdestBufflen)()
            DstBufferlen = convertdestBufflen
            
            stConvertParam.nWidth = stFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stFrame.stFrameInfo.nHeight
            stConvertParam.enDstPixelType = enDstPixelType
            stConvertParam.pDstBuffer = DstBuffer
            stConvertParam.nDstBufferSize = DstBufferlen

            ret = User.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                print ("convert pixel fail! ret[0x%x]" % ret)
                break 
        
            if nChannelNum == 1:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=DstBufferlen). \
                    reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth)
                
                # 执行1次, 只保存1张图
                if g_SaveImageOnce == False:
                    output_path = "output_mono_image.bmp"
                    cv2.imwrite(output_path, numpy_image)
                    print("save output_mono_image.bmp success")
                    
                    g_SaveImageOnce = True
            else:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=DstBufferlen). \
                    reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3)
                
                # 执行1次, 只保存1张图
                if g_SaveImageOnce == False:
                    output_path = "output_color_image.bmp"
                    cv2.imwrite(output_path, numpy_image)  
                    print("save output_color_image.bmp success")
                    
                    g_SaveImageOnce = True
            
            #正常退出
            break 
        
        #非自动释放模式，需要额外进行 手动释放资源
        if bAutoFree == False :
            if pUser != None: 
                User.MV_CC_FreeImageBuffer(stFrame)
            else :
                print ("user is null, invalid")


CALL_BACK_FUN2 = FrameInfoCallBack2(image_callback2)

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

    print ("find %d devices!" % deviceList.nDeviceNum)

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

    # ch:注册抓图回调 | en:Register image callback
    ret = cam.MV_CC_RegisterImageCallBackEx2(CALL_BACK_FUN2, ctypes.py_object(cam), True)
    if ret != 0:
        print ("register image callback fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    print ("press Enter key to stop grabbing.")
    input_func()

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
