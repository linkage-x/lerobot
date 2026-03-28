/*
* 这个示例演示线阵相机如何通过软触发命令获取图像
* This simple shows how a line-scan camera acquires images through software trigger commands.
*
*/
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include "MvCameraControl.h"

bool g_bExit = false;
char g_chTriggerCmd[128];

// ch: 检查属性节点的访问模式 | en: Check the access mode of the feature node
bool CheckFeatureNodeAccess(void* hHandle, const char* pNodeName);
// ch：打印相机信息 | en: Print the information of cameras
bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);
// ch: 取图回调函数 | en: Image callback function
void __stdcall ImageCallbackEx2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree);
// ch: 软触发命令线程 | en: software trigger command thread
static  void* SoftwareTriggerCommandThread(void* pUser);

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
    void* handle = NULL;

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
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE, &stDeviceList);
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

        printf("Please Input camera index(0-%d):", stDeviceList.nDeviceNum-1);
        unsigned int nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stDeviceList.nDeviceNum)
        {
            printf("Input error!\n");
            break;
        }

        // ch:选择设备并创建句柄 | en:Select device and create handle
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

        // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
            if (nPacketSize > 0)
            {
                nRet = MV_CC_SetIntValueEx(handle,"GevSCPSPacketSize",nPacketSize);
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

        nRet = MV_CC_SetEnumValueByString(handle, "ScanMode", "FrameScan");
        if (MV_OK == nRet)
        {
            printf("Set Frame Scan Mode \n");
        }

        // ch: 判断FrameTriggerControl是否可读 | en: Check if FrameTriggerControl is readable
        if (CheckFeatureNodeAccess(handle, "FrameTriggerControl"))
        {
            // ch:设置触发模式为on | en:Set trigger mode as on
            nRet = MV_CC_SetBoolValue(handle, "FrameTriggerMode", true);
            if (MV_OK != nRet)
            {
                printf("Set Frame Trigger Mode fail! nRet [0x%x]\n", nRet);
                break;
            }

            // ch:设置触发源为软触发 | en:Set trigger source as Software
            nRet = MV_CC_SetEnumValueByString(handle, "FrameTriggerSource", "Software");
            if (MV_OK != nRet)
            {
                printf("Set Frame Trigger source fail! nRet [0x%x]\n", nRet);
                break;
            }

            strcpy(g_chTriggerCmd, "FrameTriggerSoftware");
        }
        else
        {
            // ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
            nRet = MV_CC_SetEnumValue(handle, "TriggerSelector", 6);
            if (MV_OK != nRet)
            {
                printf("Set Trigger Selector fail! nRet [0x%x]\n", nRet);
                break;
            }

            // ch:设置触发模式为on | en:Set trigger mode as on
            nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 1);
            if (MV_OK != nRet)
            {
                printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
                break;
            }

            // ch:设置触发源为Software | en:Set trigger source as Software
            nRet = MV_CC_SetEnumValueByString(handle, "TriggerSource", "Software");
            if (MV_OK != nRet)
            {
                printf("Set Trigger source fail! nRet [0x%x]\n", nRet);
                break;
            }

            strcpy(g_chTriggerCmd, "TriggerSoftware");
        }

        // ch:注册抓图回调 | en:Register image callback
        nRet = MV_CC_RegisterImageCallBackEx2(handle, ImageCallbackEx2, handle, true);
        if (MV_OK != nRet)
        {
            printf("Register Image CallBack fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:开始取流 | en:Start grab image
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }
        
        pthread_t nThreadID;
        nRet = pthread_create(&nThreadID, NULL ,SoftwareTriggerCommandThread , handle);
        if (nRet != 0)
        {
            printf("thread create failed.ret = %d\n",nRet);
            break;
        }

        PressEnterToExit();

        g_bExit = true;
        sleep(1);

        // ch:停止取流 | en:Stop grab image
        nRet = MV_CC_StopGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:注销抓图回调 | en:Unregister image callback
        nRet = MV_CC_RegisterImageCallBackEx2(handle, NULL, NULL, true);
        if (MV_OK != nRet)
        {
            printf("Unregister Image CallBack fail! nRet [0x%x]\n", nRet);
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


    return 0;
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
        printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
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
        printf("Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
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


//ch: 检查属性节点的访问模式 | en: Check the access mode of the feature node
bool CheckFeatureNodeAccess(void* hHandle, const char* pNodeName)
{
    if (NULL == pNodeName|| NULL == hHandle)
    {
        return false;
    }

    MV_XML_AccessMode nMode;
    unsigned int nRet = MV_XML_GetNodeAccessMode(hHandle, pNodeName, &nMode);
    if (nRet != MV_OK)
    {
        return false;
    }

    if (nMode == AM_WO || nMode == AM_RO || nMode == AM_RW)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// ch: 发送软触发命令 | en: send software trigger command
static void* SoftwareTriggerCommandThread(void* pUser)
{
    int nRet = MV_OK;

    if (pUser == NULL)
    {
        return 0;
    }

    while (true)
    {
        nRet = MV_CC_SetCommandValue(pUser, g_chTriggerCmd);
        if (nRet != MV_OK)
        {
            printf("Set software trigger command fail! nRet [0x%x]\n", nRet);
        }
        
        sleep(1);
        
        if (g_bExit)
        {
            break;
        }
    }

    return 0;
}
