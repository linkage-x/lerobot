#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "MvCameraControl.h"

//ch：打印设备信息 | en: Print the information of devices
bool PrintDeviceInfo(MV_CC_DEVICE_INFO_LIST& stDeviceList);
//ch：打印采集卡信息 | en: Print the information of frame grabbers
bool PrintInterfaceInfo(MV_INTERFACE_INFO_LIST& stInterfaceInfoList);
//ch：取流回调函数 | en: Stream grabbing callback function
void __stdcall ImageCallbackEx2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree);

// 等待用户输入enter键来结束取流或结束程序
// wait for user to input enter to stop grabbing or end the sample program
void PressEnterToExit(void)
{
    int c;
    while ( (c = getchar()) != '\n' && c != EOF );
    fprintf( stderr, "\nPress enter to exit.\n");
    while( getchar() != '\n');
}



int main()
{
    int nRet = MV_OK;
    // ch:初始化SDK | en:Initialize SDK
    MV_CC_Initialize();

    //ch: 采集卡接口类型 | en: The interface type of frame grabber
    int nInterfaceType = MV_GIGE_INTERFACE | MV_CAMERALINK_INTERFACE | MV_CXP_INTERFACE | MV_XOF_INTERFACE;

    void* hInterface = NULL;  //ch: 采集卡句柄 | en: Handle of frame grabber
    void* hDevHandle = NULL;  //ch: 设备句柄 | en: Handle of device

    do 
    {
        // ch:初始化SDK | en:Initialize SDK
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet)
        {
            printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
            break;
        }

        //ch: 枚举采集卡 | en: Enumerate frame grabbers
        MV_INTERFACE_INFO_LIST stInterfaceInfoList={0};
        nRet = MV_CC_EnumInterfaces(nInterfaceType, &stInterfaceInfoList);
        if (MV_OK != nRet)
        {
            printf("Enum interfaces fail! nRet [0x%x]\n", nRet);
            break;
        }

        if (stInterfaceInfoList.nInterfaceNum > 0)
        {
            PrintInterfaceInfo(stInterfaceInfoList);
            printf("Enum interfaces success!\n\n");
        } 
        else
        {
            printf("Find no interface!\n");
            break;
        }

        printf("Please input interfaces index(0-%d):", stInterfaceInfoList.nInterfaceNum-1);
        unsigned int nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stInterfaceInfoList.nInterfaceNum)
        {
            printf("Input error!\n");
            break;
        }

        //ch:创建采集卡句柄 | en: Create the handle of frame grabber
        nRet = MV_CC_CreateInterface(&hInterface, stInterfaceInfoList.pInterfaceInfos[nIndex]);
        if (MV_OK == nRet)
        {
            printf("Create interface success!\n");
        }
        else
        {
            printf("Create interface Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        //ch: 打开采集卡 | en: Open the frame grabber 
        nRet = MV_CC_OpenInterface(hInterface, NULL);
        if (MV_OK == nRet)
        {
            printf("Open interface success!\n");
        }
        else
        {
            printf("Open interface fail! nRet [0x%x]\n", nRet);
            break;
        }

        //ch: 枚举采集卡上的相机 | en: Enumerate the devices on the frame grabber
        MV_CC_DEVICE_INFO_LIST stDeviceList = {0};
        nRet = MV_CC_EnumDevicesByInterface(hInterface, &stDeviceList);
        if (nRet != MV_OK)
        {
            printf("MV_CC_EnumDevicesByInterface fail! nRet [0x%x]\n", nRet);
            break;
        }

        if (stDeviceList.nDeviceNum == 0)
        {
            printf("no device! fail!\n");
            break;
        }

        PrintDeviceInfo(stDeviceList);

        printf("Please input camera index(0-%d):", stDeviceList.nDeviceNum-1);
        nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stDeviceList.nDeviceNum)
        {
            printf("Input error!\n");
            break;
        }

        // ch:选择设备并创建句柄 | en:Select device and create handle
        nRet = MV_CC_CreateHandle(&hDevHandle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            printf("Create handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:打开设备 | en:Open device
        nRet = MV_CC_OpenDevice(hDevHandle);
        if (MV_OK != nRet)
        {
            printf("Open device fail! nRet [0x%x]\n", nRet);
            break;
        }

        printf("Open device success!\n");

        // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(hDevHandle);
            if (nPacketSize > 0)
            {
                nRet = MV_CC_SetIntValueEx(hDevHandle,"GevSCPSPacketSize",nPacketSize);
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

        // ch:注册抓图回调 | en:Register image callback
        nRet = MV_CC_RegisterImageCallBackEx2(hDevHandle, ImageCallbackEx2, hDevHandle, true);
        if (MV_OK != nRet)
        {
            printf("Register Image CallBack fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:设置触发模式为off | en:Set trigger mode as off
        nRet = MV_CC_SetEnumValue(hDevHandle, "TriggerMode", 0);
        if (MV_OK != nRet)
        {
            printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:开始取流 | en:Start grab image
        nRet = MV_CC_StartGrabbing(hDevHandle);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        printf("Start grabbing success!\n");

        printf("Press a key to stop grabbing.\n");
        PressEnterToExit();

        //ch：停止取流 | en: Stop grabbing
        MV_CC_StopGrabbing(hDevHandle);
        printf("Stop grabbing success!\n");

        // ch:关闭设备 | Close device
        MV_CC_CloseDevice(hDevHandle);
        printf("Close device success!\n");

        // ch:销毁设备句柄 | Destroy handle of device
        MV_CC_DestroyHandle(hDevHandle);
        printf("Destroy device handle success!\n");

        hDevHandle = NULL;

        //ch: 关闭采集卡 | en: Close the frame grabber
        MV_CC_CloseInterface(hInterface);
        printf("Close interface success!\n");

        // ch:销毁采集卡句柄 | Destroy handle of frame grabber
        nRet = MV_CC_DestroyInterface(hInterface);
        printf("Destroy interface success!\n");

        hInterface = NULL;
        
    } while (0);

    if (hDevHandle != NULL)
    {
        MV_CC_CloseDevice(hDevHandle);
        MV_CC_DestroyHandle(hDevHandle);
        hDevHandle = NULL;
    }

    if (hInterface != NULL)
    {
        MV_CC_CloseInterface(hInterface);
        MV_CC_DestroyInterface(hInterface);
        hInterface = NULL;
    }
    
    // ch:反初始化SDK | en:Finalize SDK
    MV_CC_Finalize();
    printf("exit.\n");

}

bool PrintDeviceInfo(MV_CC_DEVICE_INFO_LIST& stDeviceList)
{
    for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
    {
        printf("[device %d]:\n", i);
        MV_CC_DEVICE_INFO* pstMVDevInfo = stDeviceList.pDeviceInfo[i];

        if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
        {
            int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

            // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
            printf("CurrentIp: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
            printf("UserDefinedName: %s\n\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            printf("Device Number: %d\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
        }
        else if (pstMVDevInfo->nTLayerType == MV_GENTL_GIGE_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chSerialNumber);
            printf("Model Name %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_GENTL_CAMERALINK_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chSerialNumber);
            printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stCMLInfo.chModelName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chSerialNumber);
            printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stCXPInfo.chModelName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_GENTL_XOF_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chSerialNumber);
            printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stXoFInfo.chModelName);
        }
        else
        {
            printf("Not support.\n");
        }
    }  

    return true;
}

bool PrintInterfaceInfo(MV_INTERFACE_INFO_LIST& stInterfaceInfoList)
{
    for (unsigned int i = 0; i < stInterfaceInfoList.nInterfaceNum; i++)
    {
        printf("[Interface %d]:\n", i);
        MV_INTERFACE_INFO* pstInterfaceInfo = stInterfaceInfoList.pInterfaceInfos[i];
        if (NULL == pstInterfaceInfo)
        {
            break;
        } 
        printf("Display name: %s\n",pstInterfaceInfo->chDisplayName);
        printf("Serial number: %s\n",pstInterfaceInfo->chSerialNumber);
        printf("model name: %s\n",pstInterfaceInfo->chModelName);
        printf("\n");            
    }

    return true;
}

void __stdcall ImageCallbackEx2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree)
{
    if (pstFrame)
    {
        printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n", pstFrame->stFrameInfo.nWidth, pstFrame->stFrameInfo.nHeight, pstFrame->stFrameInfo.nFrameNum);


        if (false == bAutoFree &&
            NULL != pUser) //手动释放
        {
            MV_CC_FreeImageBuffer(pUser, pstFrame);
        }
    }
    
}