#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "MvCameraControl.h"

void* g_hHandle = NULL;
bool  g_bConnect = false;
char  g_strSerialNumber[64] = {0};
bool  g_bExit = false;
// 等待用户输入enter键来结束取流或结束程序
// wait for user to input enter to stop grabbing or end the sample program
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
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        printf("CurrentIp: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }
    else
    {
        printf("Not support.\n");
    }

    return true;
}


void __stdcall ImageCallbackEx2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree)
{
    if (pstFrame)
    {
        printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n", 
            pstFrame->stFrameInfo.nExtendWidth, pstFrame->stFrameInfo.nExtendHeight, pstFrame->stFrameInfo.nFrameNum);

        if (false == bAutoFree &&
            NULL != pUser) //非自动释放模式，需要手动释放资源
        {
            MV_CC_FreeImageBuffer(pUser, pstFrame);
        }
    }
}


void __stdcall ReconnectDevice(unsigned int nMsgType, void* pUser)
{
    int nRet = MV_OK;
    if(nMsgType == MV_EXCEPTION_DEV_DISCONNECT)
    {
        
        if(true == g_bConnect)
        {
           
            nRet = MV_CC_CloseDevice(g_hHandle);
            nRet = MV_CC_DestroyHandle(g_hHandle);
            g_hHandle = NULL;
                
            MV_CC_DEVICE_INFO_LIST stDevTempList = { 0 };
            memset(&stDevTempList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
            
            unsigned int nIndex = -1;
            
            printf("device diconnect, please wait...\n");
            do 
            {
                int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDevTempList);
                if (MV_OK != nRet)
                {
                    printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
                    continue;
                }
                
                bool bFind = false;
                for (int i = 0; i< stDevTempList.nDeviceNum; i++)
                {
                
                    if (stDevTempList.pDeviceInfo[i]->nTLayerType == MV_USB_DEVICE)
                    {
                        if (0 == strcmp((char*)(stDevTempList.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber), g_strSerialNumber))
                        {
                            nIndex = i;
                            bFind = true;
                            break;
                        }
                        
                    }
                    else if (stDevTempList.pDeviceInfo[i]->nTLayerType == MV_GIGE_DEVICE)
                    {
                        if (0 == strcmp((char*)(stDevTempList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.chSerialNumber), g_strSerialNumber))
                        {
                            nIndex = i;
                            bFind = true;
                            break;
                        }
                        
                    }
                    else
                    {
                        // do nothing
                    }
                }
                if ((-1 == nIndex) || (false == bFind))
                {
                    continue;
                }
                
                // 选择设备并创建句柄
                // select device and create handle
                nRet = MV_CC_CreateHandle(&g_hHandle, stDevTempList.pDeviceInfo[nIndex]);
                if (MV_OK != nRet)
                {
                    printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
                    continue;
                }

                // 打开设备
                // open device
                nRet = MV_CC_OpenDevice(g_hHandle);
                if (MV_OK != nRet)
                {
                    printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
                    MV_CC_DestroyHandle(g_hHandle);
                    g_hHandle = NULL;
                    continue;
                }
                else
                {
                    break;  // open success退出循环
                }
                
            }while(g_bExit== false);

            
            MV_CC_RegisterExceptionCallBack(g_hHandle,ReconnectDevice, g_hHandle);

            // register image callback
            MV_CC_RegisterImageCallBackEx2(g_hHandle, ImageCallbackEx2, g_hHandle, true);
            
            nRet = MV_CC_StartGrabbing(g_hHandle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
                return;
            }

        }
    }
    
}

int main()
{
    int nRet = MV_OK;
    MV_CC_DEVICE_INFO_LIST stDeviceList = {0};
    unsigned int nSelectNum = 0;
    // ch:初始化SDK | en:Initialize SDK
    nRet = MV_CC_Initialize();
    if (MV_OK != nRet)
    {
        printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
        return nRet;
    }
    // 枚举设备
    // enum device
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
        return nRet;
    }
    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
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
        return nRet;
    }

    printf("Please Intput camera index: ");
    scanf("%d", &nSelectNum);

    if (nSelectNum >= stDeviceList.nDeviceNum)
    {
        printf("Intput error!\n");
        return nRet;
    }

    if (stDeviceList.pDeviceInfo[nSelectNum]->nTLayerType == MV_GIGE_DEVICE) 
    {
        memcpy(g_strSerialNumber, stDeviceList.pDeviceInfo[nSelectNum]->SpecialInfo.stGigEInfo.chSerialNumber, 
            sizeof(stDeviceList.pDeviceInfo[nSelectNum]->SpecialInfo.stGigEInfo.chSerialNumber));
    }
    else if (stDeviceList.pDeviceInfo[nSelectNum]->nTLayerType == MV_USB_DEVICE)
    {
        memcpy(g_strSerialNumber, stDeviceList.pDeviceInfo[nSelectNum]->SpecialInfo.stUsb3VInfo.chSerialNumber, 
            sizeof(stDeviceList.pDeviceInfo[nSelectNum]->SpecialInfo.stUsb3VInfo.chSerialNumber));
    }


    // select device and create handle
    nRet = MV_CC_CreateHandle(&g_hHandle, stDeviceList.pDeviceInfo[nSelectNum]);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
        return nRet;
    }

    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(g_hHandle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
        return nRet;
    }
    
    g_bConnect = true;
        
    // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if (stDeviceList.pDeviceInfo[nSelectNum]->nTLayerType == MV_GIGE_DEVICE)
    {
        int nPacketSize = MV_CC_GetOptimalPacketSize(g_hHandle);
        if (nPacketSize > 0)
        {
            nRet = MV_CC_SetIntValueEx(g_hHandle,"GevSCPSPacketSize",nPacketSize);
            if(nRet != MV_OK)
            {
                printf("Warning: Set Packet Size fail nRet [0x%x]!\n", nRet);
            }
        }
        else
        {
            printf("Warning: Get Packet Size fail nRet [0x%x]!\n", nPacketSize);
        }
    }
    
    // 设置触发模式为off
    // set trigger mode as off
    nRet = MV_CC_SetEnumValue(g_hHandle, "TriggerMode", 0);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
        return nRet;
    }

    // 注册异常回调
    // register Exception callback
    nRet = MV_CC_RegisterExceptionCallBack(g_hHandle, ReconnectDevice, g_hHandle);
    if (MV_OK != nRet)
    {
        printf("Register ExceptionCallBack fail! nRet [%x]\n", nRet); 
    }
    
    // 注册抓图回调
    // register image callback
    nRet = MV_CC_RegisterImageCallBackEx2(g_hHandle, ImageCallbackEx2, g_hHandle, true);
    if (MV_OK != nRet)
    {
        printf("MV_CC_RegisterImageCallBackEx fail! nRet [%x]\n", nRet);
        return nRet;
    }

    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(g_hHandle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        return nRet;
    }
    
   

    PressEnterToExit();
    g_bConnect = false;
    g_bExit = true;
    

    // 关闭设备
    // close device
    nRet = MV_CC_CloseDevice(g_hHandle);
    if (MV_OK == nRet)
    {
        printf("Close Device success! \n");
    }
    // 销毁句柄
    // destroy handle
    nRet = MV_CC_DestroyHandle(g_hHandle);
    if (MV_OK == nRet)
    {
        printf("Destroy Handle success! \n");
    }
    
    g_hHandle = NULL;
    // ch:反初始化SDK | en:Finalize SDK
    MV_CC_Finalize();
    printf("exit\n");
    return 0;
}
