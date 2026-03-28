# -- coding: utf-8 --

import sys
import os
import platform
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
    

exposure_num = 0  # 分时频闪的灯数

HB_format_list = [
    PixelType_Gvsp_HB_Mono8,
    PixelType_Gvsp_HB_Mono10,
    PixelType_Gvsp_HB_Mono10_Packed,
    PixelType_Gvsp_HB_Mono12,
    PixelType_Gvsp_HB_Mono12_Packed,
    PixelType_Gvsp_HB_Mono16,
    PixelType_Gvsp_HB_BayerGR8,
    PixelType_Gvsp_HB_BayerRG8,
    PixelType_Gvsp_HB_BayerGB8,
    PixelType_Gvsp_HB_BayerBG8,
    PixelType_Gvsp_HB_BayerRBGG8,
    PixelType_Gvsp_HB_BayerGR10,
    PixelType_Gvsp_HB_BayerRG10,
    PixelType_Gvsp_HB_BayerGB10,
    PixelType_Gvsp_HB_BayerBG10,
    PixelType_Gvsp_HB_BayerGR12,
    PixelType_Gvsp_HB_BayerRG12,
    PixelType_Gvsp_HB_BayerGB12,
    PixelType_Gvsp_HB_BayerBG12,
    PixelType_Gvsp_HB_BayerGR10_Packed,
    PixelType_Gvsp_HB_BayerRG10_Packed,
    PixelType_Gvsp_HB_BayerGB10_Packed,
    PixelType_Gvsp_HB_BayerBG10_Packed,
    PixelType_Gvsp_HB_BayerGR12_Packed,
    PixelType_Gvsp_HB_BayerRG12_Packed,
    PixelType_Gvsp_HB_BayerGB12_Packed,
    PixelType_Gvsp_HB_BayerBG12_Packed,
    PixelType_Gvsp_HB_YUV422_Packed,
    PixelType_Gvsp_HB_YUV422_YUYV_Packed,
    PixelType_Gvsp_HB_RGB8_Packed,
    PixelType_Gvsp_HB_BGR8_Packed,
    PixelType_Gvsp_HB_RGBA8_Packed,
    PixelType_Gvsp_HB_BGRA8_Packed,
    PixelType_Gvsp_HB_RGB16_Packed,
    PixelType_Gvsp_HB_BGR16_Packed,
    PixelType_Gvsp_HB_RGBA16_Packed,
    PixelType_Gvsp_HB_BGRA16_Packed]


def print_devices_info(deviceList):
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

        print_devices_info(deviceList)

        nConnectionNum = input_func("please input the number of the device to connect:")

        if int(nConnectionNum) >= deviceList.nDeviceNum:
            print("input error!")
            sys.exit()

        str_exposure_num = input_func("please input current exposure number:")
        exposure_num = int(str_exposure_num)
        if exposure_num > MV_MAX_SPLIT_NUM or exposure_num <= 1:
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
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", 0)
        if ret != 0:
            raise Exception("set trigger mode fail! ret[0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("start grabbing fail! ret[0x%x]" % ret)

        # ch:获取的帧信息 | en: frame from device
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        # ch:解码参数 | en: decode parameters
        stDecodeParam = MV_CC_HB_DECODE_PARAM()
        memset(byref(stDecodeParam), 0, sizeof(stDecodeParam))

        # ch:重构参数 | en: reconstruct image parameters
        stReconstructParam = MV_RECONSTRUCT_IMAGE_PARAM()
        memset(byref(stReconstructParam), 0, sizeof(stReconstructParam))

        # ch:重构后的图像列表 | en: image data list,after reconstruction
        dst_buffer_list = []

        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 20000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))

            # ch:如果图像是HB格式，需要先解码 | en:If the image is HB format, should to be decoded first
            if stOutFrame.stFrameInfo.enPixelType in HB_format_list:
                # 获取数据包大小
                stParam = MVCC_INTVALUE()
                memset(byref(stParam), 0, sizeof(stParam))
                ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
                if 0 != ret:
                    raise Exception("Get PayloadSize fail! ret[0x%x]" % ret)
                nPayloadSize = stParam.nCurValue
                stDecodeParam.pSrcBuf = stOutFrame.pBufAddr
                stDecodeParam.nSrcLen = stOutFrame.stFrameInfo.nFrameLen
                stDecodeParam.pDstBuf = (c_ubyte * nPayloadSize)()
                stDecodeParam.nDstBufSize = nPayloadSize
                ret = cam.MV_CC_HBDecode(stDecodeParam)
                if ret != 0:
                    cam.MV_CC_FreeImageBuffer(stOutFrame)
                    raise Exception("HB Decode fail! ret[0x%x]" % ret)
                else:
                    stReconstructParam.nWidth = stDecodeParam.nWidth
                    stReconstructParam.nHeight = stDecodeParam.nHeight
                    stReconstructParam.enPixelType = stDecodeParam.enDstPixelType
                    stReconstructParam.pSrcData = stDecodeParam.pDstBuf
                    stReconstructParam.nSrcDataLen = stDecodeParam.nDstBufLen
            else:
                stReconstructParam.nWidth = stOutFrame.stFrameInfo.nWidth
                stReconstructParam.nHeight = stOutFrame.stFrameInfo.nHeight
                stReconstructParam.enPixelType = stOutFrame.stFrameInfo.enPixelType
                stReconstructParam.pSrcData = stOutFrame.pBufAddr
                stReconstructParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen

            stReconstructParam.nExposureNum = exposure_num
            stReconstructParam.enReconstructMethod = MV_SPLIT_BY_LINE

            dst_buffer_len = int(stReconstructParam.nSrcDataLen/exposure_num)
            for i in range(exposure_num):
                dst_buffer = (c_ubyte * dst_buffer_len)()
                dst_buffer_list.append(dst_buffer)
                stReconstructParam.stDstBufList[i].pBuf = dst_buffer_list[i]
                stReconstructParam.stDstBufList[i].nBufSize = dst_buffer_len

            # ch:图像重构 | en:Image Reconstruct
            ret = cam.MV_CC_ReconstructImage(stReconstructParam)
            if ret != 0:
                cam.MV_CC_FreeImageBuffer(stOutFrame)
                raise Exception("MV_CC_ReconstructImage fail! ret[0x%x]" % ret)
            else:
                print("Reconstruct image success")

            # ch: 保持图像到文件 | en: Save image to file
            file_name = "Image_w%d_h%d_fn%d.bmp" % (stReconstructParam.nWidth, stReconstructParam.nHeight,
                                                    stOutFrame.stFrameInfo.nFrameNum)
            c_file_path = file_name.encode('ascii')
            stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            stSaveParam.enPixelType = stReconstructParam.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
            stSaveParam.nWidth = stReconstructParam.nWidth  # ch:相机对应的宽 | en:Width
            stSaveParam.nHeight = 0  # ch:相机对应的高 | en:Height0
            stSaveParam.nDataLen = 0

            # ch: 拆图会丢弃多余的行, 这里存图的高度和数据长度要重新计算
            sub_len = stReconstructParam.stDstBufList[i].nBufLen
            for i in range(exposure_num):
                stSaveParam.nDataLen = stSaveParam.nDataLen + sub_len
                stSaveParam.nHeight = stSaveParam.nHeight + stReconstructParam.stDstBufList[i].nHeight

            img_buff = (c_ubyte * stSaveParam.nDataLen)()
            for i in range(exposure_num):
                ctypes.memmove(ctypes.addressof(img_buff)+i*sub_len, stReconstructParam.stDstBufList[i].pBuf, sub_len)

            stSaveParam.pData = img_buff
            stSaveParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
            stSaveParam.pcImagePath = create_string_buffer(c_file_path)
            stSaveParam.iMethodValue = 1
            ret = cam.MV_CC_SaveImageToFileEx(stSaveParam)
            if ret != 0:
                print("Save stitched image fail! ret[0x%x]" % ret)
            else:
                print("Save stitched image success")
            cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            raise Exception("no data[0x%x]" % ret)

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
