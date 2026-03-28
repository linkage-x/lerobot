#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "MvCameraControl.h"

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
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
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
        printf("Open Device success.\n");


        printf("\n");
        printf("0: Read the file from the device and save it locally:\n");   //0: 从设备读取文件并保存到本地
        printf("1: Import the local file into the device:\n");               //1: 将本地文件导入到设备中
        printf("Please enter your choice:\n");
        unsigned int nOperationIndex = 0;
        scanf("%d", &nOperationIndex);

        if (0 != nOperationIndex && 1 != nOperationIndex)
        {
            printf("Input error!\n");
            break;
        }


        if (0 == nOperationIndex)  
        {
            //从设备读取文件并保存到本地

            //ch:1. 设备切换UserSet1 参数组    | en:1. Device switches to UserSet1 parameter set
            nRet = MV_CC_SetEnumValueByString(handle, "UserSetSelector", "UserSet1");
            if (MV_OK != nRet)
            {
                printf("Set UserSetSelector UserSet1 fail! nRet [0x%x]\n", nRet);
                break;
            }

            //ch:2. 相机加载当前组的参数  | en: 2. Camera loads parameters of the current group
            nRet = MV_CC_SetCommandValue(handle, "UserSetLoad");
            if (MV_OK != nRet)
            {
                printf("Set command UserSetLoad fail! nRet [0x%x]\n", nRet);
                break;
            }

            //ch:3. 从相机读取 "UserSet1" 信息并保存到本地 "UserSet1.bin" 文件中  ; (设备支持 "UserSet1" "UserSet2" "UserSet3"   可选）
            //en:3. Read the 'UserSet1' information from the camera and save it to the local 'UserSet1.bin' file. (The device supports 'UserSet1', 'UserSet2', and 'UserSet3' (optional).)
              
            MV_CC_FILE_ACCESS stFileAccess = { 0 };
            stFileAccess.pUserFileName = "UserSet1.bin";
            stFileAccess.pDevFileName = "UserSet1";
            nRet = MV_CC_FileAccessRead(handle, &stFileAccess);  //此接口是阻塞的，读取完成后返回， 接口耗时可能略长; (优化方式1. 可以开启单独线程进行文件写入 ;  优化方式2.可以开启单独线程使用MV_CC_GetFileAccessProgress 实时获取写入进度）
            if (MV_OK != nRet)
            {
                printf("File Access Read fail! nRet [0x%x]\n", nRet);
                break;
            }
            printf("File Access Read Success.\n");
        }
        else
        {
            // 将本地文件导入到设备中 

            //ch:1. 切换当前设备参数组为 UserSet1  | en: 1. Switch the current device parameter set to UserSet1
            nRet = MV_CC_SetEnumValueByString(handle, "UserSetSelector", "UserSet1");
            if (MV_OK != nRet)
            {
                printf("Set UserSetSelector UserSet1 fail! nRet [0x%x]\n", nRet);
                break;
            }

            //ch:2. 将本地文件"UserSet1.bin" 导入到设备 "UserSet1" 中 |en: 2. Import the local file 'UserSet1.bin' into the device 'UserSet1'.
            MV_CC_FILE_ACCESS stFileAccess = { 0 };
            stFileAccess.pUserFileName = "UserSet1.bin";
            stFileAccess.pDevFileName = "UserSet1";
            nRet = MV_CC_FileAccessWrite(handle, &stFileAccess);      //此接口是阻塞的，写入完成后返回， 接口耗时可能略长; （优化方式1. 可以开启单独线程进行文件写入 ;  优化方式2.可以开启单独线程使用MV_CC_GetFileAccessProgress 实时获取写入进度）
            if (MV_OK != nRet)
            {
                printf("File Access Write fail! nRet [0x%x]\n", nRet);
            }
            printf("File Access Write Success.\n");

            //ch:3. 相机加载当前组写入配置   | en: 3. Camera loads parameters of the current set
            nRet = MV_CC_SetCommandValue(handle, "UserSetLoad");
            if (MV_OK != nRet)
            {
                printf("Set command UserSetLoad fail! nRet [0x%x]\n", nRet);
                break;
            }

            //ch:4. 配置相机默认 以 UserSet1 参数组 启动 | en: 4. Configure the camera to start with the UserSet1 parameter set by default.
            nRet = MV_CC_SetEnumValueByString(handle, "UserSetDefault", "UserSet1");
            if (MV_OK != nRet)
            {
                printf("Set UserSetDefault UserSet1 fail! nRet [0x%x]\n", nRet);
                break;
            }

            //ch:5. 保存配置   | en: 5. Save Configuration
            nRet = MV_CC_SetCommandValue(handle, "UserSetSave");
            if (MV_OK != nRet)
            {
                printf("Set command UserSetSave fail! nRet [0x%x]\n", nRet);
                break;
            }
        }

        // ch:关闭设备 | Close device
        nRet = MV_CC_CloseDevice(handle);
        if (MV_OK != nRet)
        {
            printf("ClosDevice fail! nRet [0x%x]\n", nRet);
            break;
        }
        printf("Clos Device success.\n");
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

    printf("exit.\n");

    return 0;
}
