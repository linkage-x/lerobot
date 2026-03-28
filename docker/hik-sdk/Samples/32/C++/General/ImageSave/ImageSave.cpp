
/*
* 这个示例演示了将相机输出的图像保存为raw、jpeg、bmp、tiff、png
* This sample shows how to save raw data,jpg,bmp,tiff and png from the camera device.
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "MvCameraControl.h"

bool g_bExit = false;
void* handle = NULL;

// ch:等待按键输入 | en:Wait for key press
void PressEnterToExit(void)
{
    int c;
    while ( (c = getchar()) != '\n' && c != EOF );
    fprintf( stderr, "\nPress enter to exit.\n");
    while( getchar() != '\n');
}
bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
        printf("CurrentIp: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("UserDefinedName: %s\n\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("Device Number: %d\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
        
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_GIGE_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chSerialNumber);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_CAMERALINK_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chSerialNumber);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chModelName);
        
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chSerialNumber);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chModelName);
        
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_XOF_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chSerialNumber);
        printf("Device Type: %d\n" , pstMVDevInfo->nTLayerType);
        printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chModelName);
        
    }
    else
    {
        printf("Not support.\n");
    }
    
    return true;
}


// ch:保存图像 | en:Save Image
int SaveImage(MV_SAVE_IAMGE_TYPE enSaveImageType,MV_FRAME_OUT* stImageInfo)
{
    char chImageName[256] = {0};
    MV_CC_IMAGE stImg;
    MV_CC_SAVE_IMAGE_PARAM stSaveParams;
    memset(&stSaveParams, 0, sizeof(MV_CC_SAVE_IMAGE_PARAM));
    memset(&stImg, 0, sizeof(MV_CC_SAVE_IMAGE_PARAM));

    stImg.enPixelType = stImageInfo->stFrameInfo.enPixelType;
    stImg.nHeight = stImageInfo->stFrameInfo.nExtendHeight; 
    stImg.nWidth = stImageInfo->stFrameInfo.nExtendWidth;
    stImg.nImageLen = stImageInfo->stFrameInfo.nFrameLenEx;
    stImg.pImageBuf = stImageInfo->pBufAddr;

    stSaveParams.enImageType = enSaveImageType;
    stSaveParams.iMethodValue = 1;
    stSaveParams.nQuality = 99;
    
    if (MV_Image_Bmp == stSaveParams.enImageType)
    {
        sprintf(chImageName,  "Image_w%d_h%d_fn%03d.bmp", stImageInfo->stFrameInfo.nExtendWidth,stImageInfo->stFrameInfo.nExtendHeight, stImageInfo->stFrameInfo.nFrameNum);
    }
    else if (MV_Image_Jpeg == stSaveParams.enImageType)
    {
        sprintf(chImageName,  "Image_w%d_h%d_fn%03d.jpg", stImageInfo->stFrameInfo.nExtendWidth,stImageInfo->stFrameInfo.nExtendHeight, stImageInfo->stFrameInfo.nFrameNum);
    }
    else if (MV_Image_Tif == stSaveParams.enImageType)
    {
        sprintf(chImageName,  "Image_w%d_h%d_fn%03d.tif", stImageInfo->stFrameInfo.nExtendWidth,stImageInfo->stFrameInfo.nExtendHeight, stImageInfo->stFrameInfo.nFrameNum);
    }
    else if (MV_Image_Png == stSaveParams.enImageType)
    {
        sprintf(chImageName, "Image_w%d_h%d_fn%03d.png", stImageInfo->stFrameInfo.nExtendWidth,stImageInfo->stFrameInfo.nExtendHeight, stImageInfo->stFrameInfo.nFrameNum);
    }

    int nRet = MV_CC_SaveImageToFileEx2(handle,&stImg, &stSaveParams, (char*)chImageName);
    
    return nRet;
}

bool IsHBPixelFormat(MvGvspPixelType ePixelType)
{
    switch (ePixelType)
    {
    case PixelType_Gvsp_HB_Mono8:
    case PixelType_Gvsp_HB_Mono10:
    case PixelType_Gvsp_HB_Mono10_Packed:
    case PixelType_Gvsp_HB_Mono12:
    case PixelType_Gvsp_HB_Mono12_Packed:
    case PixelType_Gvsp_HB_Mono16:
    case PixelType_Gvsp_HB_RGB8_Packed:
    case PixelType_Gvsp_HB_BGR8_Packed:
    case PixelType_Gvsp_HB_RGBA8_Packed:
    case PixelType_Gvsp_HB_BGRA8_Packed:
    case PixelType_Gvsp_HB_RGB16_Packed:
    case PixelType_Gvsp_HB_BGR16_Packed:
    case PixelType_Gvsp_HB_RGBA16_Packed:
    case PixelType_Gvsp_HB_BGRA16_Packed:
    case PixelType_Gvsp_HB_YUV422_Packed:
    case PixelType_Gvsp_HB_YUV422_YUYV_Packed:
    case PixelType_Gvsp_HB_BayerGR8:
    case PixelType_Gvsp_HB_BayerRG8:
    case PixelType_Gvsp_HB_BayerGB8:
    case PixelType_Gvsp_HB_BayerBG8:
    case PixelType_Gvsp_HB_BayerRBGG8:
    case PixelType_Gvsp_HB_BayerGB10:
    case PixelType_Gvsp_HB_BayerGB10_Packed:
    case PixelType_Gvsp_HB_BayerBG10:
    case PixelType_Gvsp_HB_BayerBG10_Packed:
    case PixelType_Gvsp_HB_BayerRG10:
    case PixelType_Gvsp_HB_BayerRG10_Packed:
    case PixelType_Gvsp_HB_BayerGR10:
    case PixelType_Gvsp_HB_BayerGR10_Packed:
    case PixelType_Gvsp_HB_BayerGB12:
    case PixelType_Gvsp_HB_BayerGB12_Packed:
    case PixelType_Gvsp_HB_BayerBG12:
    case PixelType_Gvsp_HB_BayerBG12_Packed:
    case PixelType_Gvsp_HB_BayerRG12:
    case PixelType_Gvsp_HB_BayerRG12_Packed:
    case PixelType_Gvsp_HB_BayerGR12:
    case PixelType_Gvsp_HB_BayerGR12_Packed:
        return true;
    default:
        return false;
    }
}

int main()
{
    int nRet = MV_OK;
    unsigned char * pDstBuf = NULL;  // 用于HB解码

    do 
    {
        // ch:初始化SDK | en:Initialize SDK
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet)
        {
            printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:枚举设备 | en:Enum device
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE , &stDeviceList);
        if (MV_OK != nRet)
        {
            printf("Enum Devices fail! nRet [0x%x]\n", nRet);
            break;
        }

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

         printf("Please Intput camera index: ");
        unsigned int nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stDeviceList.nDeviceNum)
        {
            printf("Input error!\n");
            break;
        }

        // ch:选择设备并创建句柄  | en:Select device and create handle
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

        // ch:̽探测网络最佳包大小(只对GigE相机有效)| en:Detection network optimal package size(It only works for the GigE camera)
        if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
            if (nPacketSize > 0)
            {
                nRet = MV_CC_SetIntValue(handle,"GevSCPSPacketSize",nPacketSize);
                if(nRet != MV_OK)
                {
                    printf("Warning: Set Packet Size fail nRet [0x%x]!", nRet);
                }
            }
            else
            {
                printf("Warning: Get Packet Size fail nRet [0x%x]!", nPacketSize);
            }
        }

        // ch:获取数据包大小 | en:Get payload size
        MVCC_INTVALUE stParam;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet)
        {
            printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
            break;
        }
        unsigned int nPayloadSize = stParam.nCurValue;

        // ch:设置触发模式off | en:Set trigger mode as off
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
        if (MV_OK != nRet)
        {
            printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:开始取流| en:Start grab image
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        MV_FRAME_OUT stImageInfo = {0};
        MV_CC_HB_DECODE_PARAM stDecodeParam = {0};
        nRet = MV_CC_GetImageBuffer(handle, &stImageInfo, 20000); //线阵相机需要更大的图像获取间隔
        if (nRet == MV_OK)
        {
            printf("Get Image Buffer: Width[%d], Height[%d], FrameNum[%d]\n",
                stImageInfo.stFrameInfo.nExtendWidth, stImageInfo.stFrameInfo.nExtendHeight, stImageInfo.stFrameInfo.nFrameNum);

           printf("Please Input Save Image Type(raw/Jpeg/bmp/tiff/png):");
            while (true)
            {    
                char chSaveImageType[128] ={0};
                scanf("%s", chSaveImageType);
                if (0 == strcmp(chSaveImageType,"raw"))
                {
                    // 判断是否为HB模式，若是进行解码
                    bool bflag = IsHBPixelFormat(stImageInfo.stFrameInfo.enPixelType);
                    if(bflag)
                    {
                        stDecodeParam.pSrcBuf = stImageInfo.pBufAddr;
                        stDecodeParam.nSrcLen = stImageInfo.stFrameInfo.nFrameLenEx;

                        if (pDstBuf == NULL)
                        {
                            pDstBuf = (unsigned char *)malloc(sizeof(unsigned char) * (nPayloadSize*3));
                            if (NULL == pDstBuf)
                            {
                                printf("malloc pDstData fail !\n");
                                nRet = MV_E_RESOURCE;
                                break;
                            }
                        }
                        stDecodeParam.pDstBuf = pDstBuf;
                        stDecodeParam.nDstBufSize = nPayloadSize;
                        nRet = MV_CC_HB_Decode(handle, &stDecodeParam);
                        if (nRet != MV_OK)
                        {
                            printf("Decode fail![0x%x]\n", nRet);
                            break;
                        }
                        else
                        {
                            printf("HB Decode success!\n");
                        }

                        // 保存裸图
                        char chRawPath[256] = {0};
                        sprintf(chRawPath,  "Image_w%d_h%d_fn%03d.raw", stDecodeParam.nWidth, stDecodeParam.nHeight, stImageInfo.stFrameInfo.nFrameNum);
                        FILE* file = fopen(chRawPath, "wb+");
                        if (NULL != file) 
                        {
                            // 写入数据到文件
                            size_t bytesWritten = fwrite(stDecodeParam.pDstBuf, 1, stDecodeParam.nDstBufLen, file);
                            if (stDecodeParam.nDstBufLen == bytesWritten)
                            {
                                printf("Save image raw success.\n");
                            }
                            else
                            {
                                printf("Save image raw failed.\n");
                            }

                            //关闭文件
                            fclose(file);
                        }
                        
                    }
                    else
                    {
                        // 保存裸图
                        char chRawPath[256] = {0};
                        sprintf(chRawPath,  "Image_w%d_h%d_fn%03d.raw", stImageInfo.stFrameInfo.nExtendWidth, stImageInfo.stFrameInfo.nExtendHeight, stImageInfo.stFrameInfo.nFrameNum);
                        FILE* file = fopen(chRawPath, "wb+");
                        if (NULL != file) 
                        {
                            // 写入数据到文件
                            size_t bytesWritten = fwrite(stImageInfo.pBufAddr, 1, stImageInfo.stFrameInfo.nFrameLenEx, file);

                            if (stImageInfo.stFrameInfo.nFrameLenEx == bytesWritten)
                            {
                                printf("Save image raw success.\n");
                            }
                            else
                            {
                                printf("Save image raw failed.\n");
                            }
                            // 关闭文件
                            fclose(file);
                        }
                    }
                    
                    break;
                }
                else if(0 == strcmp(chSaveImageType,"Jpeg"))
                {
                    nRet = SaveImage(MV_Image_Jpeg,&stImageInfo);
                    if(MV_OK == nRet)
                    {
                        printf("Save Image Jpeg success.\n");
                    }
                    else
                    {
                        printf("Save image Jpeg failed.\n");
                    }
                    break;
                }
                else if(0 == strcmp(chSaveImageType,"bmp"))
                {
                    nRet = SaveImage(MV_Image_Bmp,&stImageInfo);
                    if(MV_OK == nRet)
                    {
                        printf("Save Image bmp success.\n");
                    }
                    else
                    {
                        printf("Save image bmp failed.\n");
                    }
                    break;
                }
                else if(0 == strcmp(chSaveImageType,"tiff"))
                {
                    nRet = SaveImage(MV_Image_Tif,&stImageInfo);
                    if(MV_OK == nRet)
                    {
                        printf("Save Image tif success.\n");
                    }
                    else
                    {
                        printf("Save image tif failed.\n");
                    }
                    break;
                }
                else if(0 == strcmp(chSaveImageType,"png"))
                {
                    nRet = SaveImage(MV_Image_Png,&stImageInfo);
                    if(MV_OK == nRet)
                    {
                        printf("Save Image png success.\n");
                    }
                    else
                    {
                        printf("Save image png failed.\n");
                    }
                    break;
                }
                else
                {
                    printf("Input not supoort,Please Input Save Image Type(raw/Jpeg/bmp/tiff/png):");
                    continue;
                }
            }
            
            if (MV_OK != nRet)
            {
                if(NULL != pDstBuf)
                {
                    free (pDstBuf);
                    pDstBuf = NULL;
                }    
            }

            nRet = MV_CC_FreeImageBuffer(handle, &stImageInfo);
            if(nRet != MV_OK)
            {
                printf("Free Image Buffer fail! nRet [0x%x]\n", nRet);
            }
            
        }
        else
        {
            printf("Get Image fail! nRet [0x%x]\n", nRet);
        }

        // ch:ֹͣ停止取流 | en:Stop grab image
        nRet = MV_CC_StopGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:关闭设备 | Close device
        nRet = MV_CC_CloseDevice(handle);
        if (MV_OK != nRet)
        {
            printf("ClosDevice fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:销毁句柄 | Destroy handle
        nRet = MV_CC_DestroyHandle(handle);
        if (MV_OK != nRet)
        {
            printf("Destroy Handle fail! nRet [0x%x]\n", nRet);
            break;
        }
        handle = NULL;
    } while (0);
 
    if (handle != NULL)
    {
        MV_CC_DestroyHandle(handle);
        handle = NULL;
    }
    

    // ch:反初始化SDK | en:Finalize SDK
    MV_CC_Finalize();

    PressEnterToExit();

    return 0;
}
