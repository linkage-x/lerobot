/*
 * 这个示例演示了如何使用采集卡快速软触发功能
 * [注] 快速软触发功能需要采集卡固件支持，目前只有部分XoF(GS1104F)和CXP(GX1002,GX1004)采集卡支持
 * This program shows how to use the quick software trigger of the frame grabbers. 
 * [PS] The quick soft trigger function requires firmware support from the frame grabber,
 *      and currently only some XoF (GS1104F) and CXP (GX1002, GX1004) frame grabbers support it.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "MvCameraControl.h"

void* g_hInterface = NULL;
bool g_bExit = false;

// 等待用户输入enter键来结束取流或结束程序
// wait for user to input enter to stop grabbing or end the sample program
void PressEnterToExit(void)
{
    int c;
    while ( (c = getchar()) != '\n' && c != EOF );
    fprintf( stderr, "\nPress enter to exit.\n");
    while( getchar() != '\n');
    g_bExit = true;
    sleep(1);
}

bool PrintInterfaceInfo(MV_INTERFACE_INFO* pstInterfaceInfo)
{
    if (NULL == pstInterfaceInfo)
    {
        printf("The Pointer of pstInterfaceInfo is NULL!\n");
        return false;
    }
    printf("Display name: %s\n",pstInterfaceInfo->chDisplayName);
    printf("Serial number: %s\n",pstInterfaceInfo->chSerialNumber);
    printf("model name: %s\n",pstInterfaceInfo->chModelName);
    printf("\n");

    return true;
}

bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
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

static void* WorkThread(void* pUser)
{
    int nRet = MV_OK;
    MV_FRAME_OUT stOutFrame = {0};

    while(true)
    {
        // ch:通过采集卡软触发一次 | en:Software trigger once from interface
        nRet = MV_CC_SetCommandValue(g_hInterface, "QuickSoftwareTrigger0");
        if (MV_OK != nRet)
        {
            printf("Quick Software Trigger once failed! %#x\n", nRet);
        }
        else
        {
            printf("Quick Software Trigger once success!\n");
        }
        nRet = MV_CC_GetImageBuffer(pUser, &stOutFrame, 1000);
        if (nRet == MV_OK)
        {
            printf("Get Image Buffer: Width[%d], Height[%d], FrameNum[%d]\n",
                stOutFrame.stFrameInfo.nExtendWidth, stOutFrame.stFrameInfo.nExtendHeight, stOutFrame.stFrameInfo.nFrameNum);

            nRet = MV_CC_FreeImageBuffer(pUser, &stOutFrame);
            if(nRet != MV_OK)
            {
                printf("Free Image Buffer fail! nRet [0x%x]\n", nRet);
            }
        }
        else
        {
            printf("Get Image fail! nRet [0x%x]\n", nRet);
        }
        if(g_bExit)
        {
            break;
        }
    }

    return 0;
}

int main()
{
    int nRet = MV_OK;
    void* hDevice = NULL;

    do 
    {
        // ch:初始化SDK | en:Initialize SDK
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet)
        {
            printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
            break;
        }

        MV_INTERFACE_INFO_LIST stInterfaceInfoList={0};
        nRet = MV_CC_EnumInterfaces(MV_CXP_INTERFACE | MV_XOF_INTERFACE, &stInterfaceInfoList);
        //枚举采集卡
        if (MV_OK != nRet)
        {
            printf("Enum Interfaces fail! nRet [0x%x]\n", nRet);
            break;
        }
        
        if (stInterfaceInfoList.nInterfaceNum > 0)
        {
            for (unsigned int i = 0; i < stInterfaceInfoList.nInterfaceNum; i++)
            {
                printf("[Interface %d]:\n", i);
                MV_INTERFACE_INFO* pstInterfaceInfo = stInterfaceInfoList.pInterfaceInfos[i];
                if (NULL == pstInterfaceInfo)
                {
                    break;
                } 
                PrintInterfaceInfo(pstInterfaceInfo);            
            }
            printf("Enum Interfaces success!\n\n");
        } 
        else
        {
            printf("Find No Interface!\n");
            break;
        }

        printf("Please Input Interfaces index(0-%d):", stInterfaceInfoList.nInterfaceNum-1);
        unsigned int nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stInterfaceInfoList.nInterfaceNum)
        {
            printf("Input error!\n");
            break;
        }

        //创建采集卡句柄
        nRet = MV_CC_CreateInterface(&g_hInterface, stInterfaceInfoList.pInterfaceInfos[nIndex]);
        if (MV_OK == nRet)
        {
            printf("Create Interface success!\n");
        }
        else
        {
            printf("Create Interface Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        //打开采集卡
        nRet = MV_CC_OpenInterface(g_hInterface, NULL);
        if (MV_OK == nRet)
        {
            printf("Open Interface success!\n");
        }
        else
        {
            printf("Open Interface fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:设置数据流触发源为快速软触发0 | en:Set Stream Trigger Source to QuickSoftwareTrigger0
        nRet = MV_CC_SetEnumValueByString(g_hInterface, "StreamTriggerSource", "QuickSoftwareTrigger0");
        if (MV_OK != nRet)
        {
            printf("Set StreamTriggerSource fail! nRet [0x%x], maybe firmware not support Quick Software Trigger\n", nRet);
            break;
        }
        else
        {
            printf("Set StreamTriggerSource = QuickSoftwareTrigger0 Success!\n");
        }

        // ch:设置数据流触发方式为上升沿 | en:Set Stream Trigger Activation to RisingEdge
        nRet = MV_CC_SetEnumValueByString(g_hInterface, "StreamTriggerActivation", "RisingEdge");
        if (MV_OK != nRet)
        {
            printf("Set StreamTriggerActivation Fail! nRet [0x%x]\n", nRet);
            break;
        }
        else
        {
            printf("Set StreamTriggerActivation = RisingEdge Success!\n");
        }

        MV_CC_DEVICE_INFO_LIST stDeviceList = { 0 };
        nRet = MV_CC_EnumDevices(MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE, &stDeviceList);
        //枚举采集卡设备
        if (MV_OK != nRet)
        {
            printf("Enum Interfaces Devices fail! nRet [0x%x]\n", nRet);
            break;
        }
        int nDeviceNum = 0;
        int nDeviceIndex[MV_MAX_DEVICE_NUM] = { 0 };
        if (stDeviceList.nDeviceNum > 0)
        {
            for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
                if (NULL == pDeviceInfo)
                {
                    break;
                }
                if (MV_CXP_INTERFACE == stInterfaceInfoList.pInterfaceInfos[nIndex]->nTLayerType)
                {
                    if (0 == strcmp((char*)stInterfaceInfoList.pInterfaceInfos[nIndex]->chInterfaceID, 
                        (char*)pDeviceInfo->SpecialInfo.stCXPInfo.chInterfaceID))
                    {
                        printf("[device %d]:\n", nDeviceNum);
                        nDeviceIndex[nDeviceNum] = i;
                        PrintDeviceInfo(pDeviceInfo);
                        nDeviceNum++;
                    }
                }
                else
                {
                    if (0 == strcmp((char*)stInterfaceInfoList.pInterfaceInfos[nIndex]->chInterfaceID, 
                        (char*)pDeviceInfo->SpecialInfo.stXoFInfo.chInterfaceID))
                    {
                        printf("[device %d]:\n", nDeviceNum);
                        nDeviceIndex[nDeviceNum] = i;
                        PrintDeviceInfo(pDeviceInfo);
                        nDeviceNum++;
                    }
                }
            }
        }
        else
        {
            printf("Find No Devices!\n");
            break;
        }

        if (0 == nDeviceNum)
        {
            printf("Find No Devices!\n");
            break;
        }

        printf("Please Input camera index(0-%d):", nDeviceNum - 1);
        nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= nDeviceNum)
        {
            printf("Input error!\n");
            break;
        }

        // ch:选择设备并创建句柄 | en:Select device and create handle
        nRet = MV_CC_CreateHandle(&hDevice, stDeviceList.pDeviceInfo[nDeviceIndex[nIndex]]);
        if (MV_OK != nRet)
        {
            printf("Create Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:打开设备 | en:Open device
        nRet = MV_CC_OpenDevice(hDevice);
        if (MV_OK != nRet)
        {
            printf("Open Device fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:设置触发模式为off | en:Set trigger mode as off
        nRet = MV_CC_SetEnumValue(hDevice, "TriggerMode", 0);
        if (MV_OK != nRet)
        {
            printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:开始取流 | en:Start grab image
        nRet = MV_CC_StartGrabbing(hDevice);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        pthread_t nThreadID;
        nRet = pthread_create(&nThreadID, NULL ,WorkThread , hDevice);
        if (nRet != 0)
        {
            printf("thread create failed.ret = %d\n",nRet);
            break;
        }

        printf("Press a key to stop grabbing.\n");
        PressEnterToExit();

        g_bExit = true;

        // ch:停止取流 | en:Stop grab image
        nRet = MV_CC_StopGrabbing(hDevice);
        if (MV_OK != nRet)
        {
            printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:关闭设备 | Close device
        nRet = MV_CC_CloseDevice(hDevice);
        if (MV_OK != nRet)
        {
            printf("ClosDevice fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:销毁句柄 | Destroy handle
        nRet = MV_CC_DestroyHandle(hDevice);
        if (MV_OK != nRet)
        {
            printf("Destroy Handle fail! nRet [0x%x]\n", nRet);
            break;
        }
        hDevice = NULL;

        //关闭采集卡
        nRet = MV_CC_CloseInterface(g_hInterface);
        if (MV_OK == nRet)
        {
            printf("Close Interface success!\n");
        }
        else
        {
            printf("Close Interface Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        //销毁采集卡句柄
        nRet = MV_CC_DestroyInterface(g_hInterface);
        if (MV_OK == nRet)
        {
            printf("Destroy Interface success!\n");
        }
        else
        {
            printf("Destroy Interface Handle fail! nRet [0x%x]\n", nRet);
            break;
        }
        g_hInterface = NULL;
    } while (0);

    if (hDevice != NULL)
    {
        MV_CC_CloseDevice(hDevice);
        MV_CC_DestroyHandle(hDevice);
        hDevice = NULL;
    }

    if (g_hInterface != NULL)
    {
        MV_CC_CloseInterface(g_hInterface);
        MV_CC_DestroyInterface(g_hInterface);
        g_hInterface = NULL;
    }

    // ch:反初始化SDK | en:Finalize SDK
    MV_CC_Finalize();

    printf("Press a key to exit.\n");
    PressEnterToExit();
    return 0;
}