/***************************************************************************************************
* 
* Notes about how to configure your OpenCV environment and project.
* 1. You can prepare the required installation package from the official website. https://opencv.org/releases.html
* 2. If the *.lib files doesn't exist in the package download, you need to compile by yourself with the CMake tool.
* 3. Add the 'bin' folder path to the PATH.
* 4. Configure the 'Additional Include Directories', 'Additional Library Directories' and 'Additional Dependencies' for current project property.
* 
* If there is any question or request, please feel free to contact us.

***************************************************************************************************/

#include <stdio.h>
#include <string.h>
#include "opencv2/core/core_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "MvCameraControl.h"

enum CONVERT_TYPE
{
    OpenCV_Mat         = 0,    // ch:Mat图像格式 | en:Mat format
    OpenCV_IplImage    = 1,    // ch:IplImage图像格式 | en:IplImage format
};

// ch:显示枚举到的设备信息 | en:Print the discovered devices' information
void PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("    NULL info.\n\n");
        return;
    }

    // 获取图像数据帧仅支持GigE和U3V设备
    if (MV_GIGE_DEVICE == pstMVDevInfo->nTLayerType)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:显示IP和设备名 | en:Print current ip and user defined name
        printf("    IP: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
        printf("    UserDefinedName: %s\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
		printf("    Device Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
    }
    else if (MV_USB_DEVICE == pstMVDevInfo->nTLayerType)
    {
        printf("    UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
		printf("    Device Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
    }
    else
    {
        printf("    Not support.\n\n");
    }
}

// ch:像素排列由RGB转为BGR | en:Convert pixel arrangement from RGB to BGR
void RGB2BGR( unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight )
{
    if ( NULL == pRgbData )
    {
        return;
    }

    // red和blue数据互换
    for (unsigned int j = 0; j < nHeight; j++)
    {
        for (unsigned int i = 0; i < nWidth; i++)
        {
            unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
            pRgbData[j * (nWidth * 3) + i * 3]     = pRgbData[j * (nWidth * 3) + i * 3 + 2];
            pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
        }
    }
}

// ch:帧数据转换为Mat格式图片并保存 | en:Convert data stream to Mat format then save image
bool Convert2Mat(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char * pData)
{
    if (NULL == pstImageInfo || NULL == pData)
    {
        printf("NULL info or data.\n");
        return false;
    }

    cv::Mat srcImage;

    if ( PixelType_Gvsp_Mono8 == pstImageInfo->enPixelType )                // Mono8类型
    {
        srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC1, pData);
    }
    else if ( PixelType_Gvsp_RGB8_Packed == pstImageInfo->enPixelType )     // RGB8类型
    {
        // Mat像素排列格式为BGR，需要转换
        RGB2BGR(pData, pstImageInfo->nWidth, pstImageInfo->nHeight);
        srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC3, pData);
    }
    else
    {
		/* Bayer 格式转换mat格式的方法:
		1. 使用相机句柄销毁前 调用 MV_CC_ConvertPixelType 将PixelType_Gvsp_BayerRG8 等Bayer格式转换成 PixelType_Gvsp_BGR8_Packed 
		2. 参考上面 将BGR转换为 mat格式		
		*/

        printf("Unsupported pixel format\n");
        return false;
    }

    if ( NULL == srcImage.data )
    {
        printf("Creat Mat failed.\n");
        return false;
    }

    try 
    {
        // ch:保存Mat图片 | en:Save converted image in a local file
        cv::imwrite("Image_Mat.bmp", srcImage);
    }
    catch (cv::Exception& ex) 
    {
        fprintf(stderr, "Exception in saving mat image: %s\n", ex.what());
    }

    srcImage.release();

    return true;
}

// ch:帧数据转换为IplImage格式图片并保存 | en:Convert data stream in Ipl format then save image
bool Convert2Ipl(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char * pData)
{
    if (NULL == pstImageInfo || NULL == pData)
    {
        printf("NULL info or data.\n");
        return false;
    }

    IplImage* srcImage = NULL;

    if ( PixelType_Gvsp_Mono8 == pstImageInfo->enPixelType )                // Mono8类型
    {
        srcImage = cvCreateImage(cvSize(pstImageInfo->nWidth, pstImageInfo->nHeight), IPL_DEPTH_8U, 1);
    }
    else if ( PixelType_Gvsp_RGB8_Packed == pstImageInfo->enPixelType )     // RGB8类型
    {
        // IplImage像素排列格式为BGR，需要转换
        RGB2BGR(pData, pstImageInfo->nWidth, pstImageInfo->nHeight);
        srcImage = cvCreateImage(cvSize(pstImageInfo->nWidth, pstImageInfo->nHeight), IPL_DEPTH_8U, 3);
    }
    else
    {
        printf("Unsupported pixel format\n");
        return false;
    }

    if ( NULL == srcImage )
    {
        printf("Creat IplImage failed.\n");
        return false;
    }

    srcImage->imageData = (char *)pData;

    try 
    {
        // ch:保存IplImage图片 | en:Save converted image in a local file
        cv::Mat cConvertImage = cv::cvarrToMat(srcImage);
        cv::imwrite("Image_Ipl.bmp", cConvertImage);

        cConvertImage.release();
    }
    catch (cv::Exception& ex) 
    {
        fprintf(stderr, "Exception in saving IplImage: %s\n", ex.what());
    }

    cvReleaseImage(&srcImage);

    return true;
}

int main()
{
    int nRet = MV_OK;
    void* handle = NULL;
    unsigned char * pData = NULL;

    do
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        // ch:设备枚举 | en:Enum device
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet)
        {
            printf("Enum Devices fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:显示设备信息 | en:Show devices
        if (stDeviceList.nDeviceNum > 0)
        {
            for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                printf("[device %d]:\n", i);
                MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
                if (NULL == pDeviceInfo)
                {
                    break;
                }
                PrintDeviceInfo(pDeviceInfo);
            }
        }
        else
        {
            printf("Find No Devices!\n");
            break;
        }

        // ch:选择相机 | en:Select device
        unsigned int nIndex = 0;
        while (1)
        {
            printf("Please Input camera index(0-%d): ", stDeviceList.nDeviceNum - 1);

            if (1 == scanf("%d", &nIndex))
            {
                while (getchar() != '\n')
                {
                    ;
                }

                // 合法输入
                if (nIndex >= 0 && nIndex < stDeviceList.nDeviceNum)
                {
                    // 设备不可连接，重新输入
                    if (false == MV_CC_IsDeviceAccessible(stDeviceList.pDeviceInfo[nIndex], MV_ACCESS_Exclusive))
                    {
                        printf("Can't connect! ");
                        continue;
                    }

                    break;
                }
            }
            else
            {
                while (getchar() != '\n')
                {
                    ;
                }
            }
        }

        // ch:创建设备句柄 | en:Create handle
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            printf("Create Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:打开设备 | en:Open device
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet)
        {
            printf("Open Device fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:探测最佳Packet大小（只支持GigE相机） | en:Detection network optimal package size(It only works for the GigE camera)
        if (MV_GIGE_DEVICE == stDeviceList.pDeviceInfo[nIndex]->nTLayerType)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
            if (nPacketSize > 0)
            {
                // 设置Packet大小
                nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
                if (MV_OK != nRet)
                {
                    printf("Warning: Set Packet Size fail! nRet [0x%x]!", nRet);
                }
            }
            else
            {
                printf("Warning: Get Packet Size fail! nRet [0x%x]!", nPacketSize);
            }
        }

        // ch:关闭触发模式 | en:Set trigger mode as off
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
        if (MV_OK != nRet)
        {
            printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:获取图像大小 | en:Get payload size
        MVCC_INTVALUE stParam;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet)
        {
            printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
            break;
        }
        unsigned int nPayloadSize = stParam.nCurValue;

        // ch:初始化图像信息 | en:Init image info
        MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        pData = (unsigned char *)malloc(sizeof(unsigned char)* (nPayloadSize));
        if (NULL == pData)
        {
            printf("Allocate memory failed.\n");
            break;
        }
        memset(pData, 0, sizeof(pData));

        // ch:开始取流 | en:Start grab image
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:获取一帧图像，超时时间1000ms | en:Get one frame from camera with timeout=1000ms
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nPayloadSize, &stImageInfo, 1000);
        if (MV_OK == nRet)
        {
            printf("Get One Frame: Width[%d], Height[%d], FrameNum[%d]\n",
                stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
        }
        else
        {
            printf("Get Frame fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:停止取流 | en:Stop grab image
        nRet = MV_CC_StopGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:关闭设备 | en:Close device
        nRet = MV_CC_CloseDevice(handle);
        if (MV_OK != nRet)
        {
            printf("ClosDevice fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:输入要转换的格式 | en:Input the format to convert
        printf("\n[0] OpenCV_Mat\n");
        printf("[1] OpenCV_IplImage\n");
        int nFormat = 0;
        while (1)
        {
            printf("Please Input Format to convert: ");

            if (1 == scanf("%d", &nFormat))
            {
                // 合法输入
                if (0 == nFormat || 1 == nFormat)
                {
                    break;
                }
            }
            while (getchar() != '\n')
            {
                ;
            }
        }

        // ch:数据转换 | en:Convert image data
        bool bConvertRet = false;
        if (OpenCV_Mat == nFormat)
        {
            bConvertRet = Convert2Mat(&stImageInfo, pData);
        }
        else if (OpenCV_IplImage == nFormat)
        {
            bConvertRet = Convert2Ipl(&stImageInfo, pData);
        }

        // ch:显示转换结果 | en:Print result
        if (bConvertRet)
        {
            printf("OpenCV format convert finished.\n");
        }
        else
        {
            printf("OpenCV format convert failed.\n");
        }
    } while (0);

    // ch:销毁句柄 | en:Destroy handle
    if (handle)
    {
        MV_CC_DestroyHandle(handle);
        handle = NULL;
    }

    // ch:释放内存 | en:Free memery
    if (pData)
    {
        free(pData);
        pData = NULL;
    }

    return 0;
}
