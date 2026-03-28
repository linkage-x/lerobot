#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "MvCameraControl.h"

void* hInterface = NULL;
//ch: 事件回调函数 | en: Event callback function
void __stdcall EventCallBack(MV_EVENT_OUT_INFO * pEventInfo, void* pUser)
{
	if (pEventInfo)
	{
		int64_t nTimestamp = pEventInfo->nTimestampHigh;
		nTimestamp = (nTimestamp << 32) + pEventInfo->nTimestampLow;

		printf("EventName[%s], EventID[%u], Timestamp[%ld]\n",
			pEventInfo->EventName, pEventInfo->nEventID, nTimestamp);
	}
}

// ch:等待按键输入 | en:Wait for key press
void PressEnterToExit(void)
{
    int c;
    while ( (c = getchar()) != '\n' && c != EOF );
    fprintf( stderr, "\nPress enter to exit.\n");
    while( getchar() != '\n');
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

void Set_Get_Enum(const char* str)
{
	MVCC_ENUMVALUE stEnumValue = {0};
	MVCC_ENUMENTRY stEnumentryInfo = {0};

	int nRet = MV_CC_GetEnumValue(hInterface,str, &stEnumValue);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str, nRet);
		return;
	}

	stEnumentryInfo.nValue = stEnumValue.nCurValue;
	nRet = MV_CC_GetEnumEntrySymbolic(hInterface,str, &stEnumentryInfo);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
		return;
	}
	else
	{
		printf("Get %s = [%s] Success!\n",str,stEnumentryInfo.chSymbolic);
	}

    MV_XML_AccessMode enAccessMode = AM_NI;
    nRet = MV_XML_GetNodeAccessMode(hInterface, str, &enAccessMode);
    if(MV_OK == nRet && AM_RW == enAccessMode)
    {
        nRet = MV_CC_SetEnumValue(hInterface,str,stEnumValue.nCurValue);
        if (MV_OK != nRet)
        {
            printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
            return;
        }
        else
        {
            printf("Set %s = [%s] Success!\n",str,stEnumentryInfo.chSymbolic);
        }
    }
}

void Set_Get_Bool(const char* str)
{
	bool bValue = false;
	int nRet = MV_CC_GetBoolValue(hInterface,str, &bValue);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
		return;
	}
	else
	{
		printf("Get %s =  [%d] Success!\n",str,bValue);
	}

    MV_XML_AccessMode enAccessMode = AM_NI;
    nRet = MV_XML_GetNodeAccessMode(hInterface, str, &enAccessMode);
    if(MV_OK == nRet && AM_RW == enAccessMode)
    {
	    nRet = MV_CC_SetBoolValue(hInterface,str, bValue);
	    if (MV_OK != nRet)
	    {
		    printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
		    return;
	    }
	    else
	    {
		    printf("Set %s = [%d] Success!\n",str,bValue);
	    }
    }
}

void Set_Get_Int(const char* str)
{
	MVCC_INTVALUE_EX stIntValue;
	int nRet = MV_CC_GetIntValueEx(hInterface,str,&stIntValue);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
		return;
	}
	else
	{
		printf("Get %s =  [%ld] Success!\n",str,stIntValue.nCurValue);
	}
	
    MV_XML_AccessMode enAccessMode = AM_NI;
    nRet = MV_XML_GetNodeAccessMode(hInterface, str, &enAccessMode);
    if(MV_OK == nRet && AM_RW == enAccessMode)
    {
        nRet = MV_CC_SetIntValueEx(hInterface,str,stIntValue.nCurValue);
        if (MV_OK != nRet)
        {
	        printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
	        return;
        }
        else
        {
	        printf("Set %s = [%ld] Success!\n",str,stIntValue.nCurValue);
        
	}
    }
}

void Set_Get_String(const char* str)
{
	MVCC_STRINGVALUE StringValue;
	int nRet = MV_CC_GetStringValue(hInterface,str, &StringValue);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
		return;
	}
	else
	{
		printf("Get %s =  [%s] Success!\n",str,StringValue.chCurValue);
	}

    MV_XML_AccessMode enAccessMode = AM_NI;
    nRet = MV_XML_GetNodeAccessMode(hInterface, str, &enAccessMode);
    if(MV_OK == nRet && AM_RW == enAccessMode)
    {
        nRet = MV_CC_SetStringValue(hInterface,str, StringValue.chCurValue);
        if (MV_OK != nRet)
        {
	        printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
	        return;
        }
        else
        {
	        printf("Set %s = [%s] Success!\n",str,StringValue.chCurValue);
        }
    }
}

void Set_Get_Float(const char* str)
{
	MVCC_FLOATVALUE FloatValue;
	int nRet = MV_CC_GetFloatValue(hInterface,str, &FloatValue);
	if (MV_OK != nRet)
	{
		printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
		return;
	}
	else
	{
		printf("Get %s =  [%f] Success!\n",str,FloatValue.fCurValue);
	}

    MV_XML_AccessMode enAccessMode = AM_NI;
    nRet = MV_XML_GetNodeAccessMode(hInterface, str, &enAccessMode);
    if(MV_OK == nRet && AM_RW == enAccessMode)
    {
        nRet = MV_CC_SetFloatValue(hInterface,str, FloatValue.fCurValue);
        if (MV_OK != nRet)
        {
	        printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
	        return;
        }
        else
        {
	        printf("Set %s = [%f] Success!\n",str,FloatValue.fCurValue);
        }
    }
}

int main()
{
	int nRet = MV_OK;

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

		// 枚举光源控制卡
		nRet = MV_CC_EnumInterfaces(MV_LC_INTERFACE, &stInterfaceInfoList);
		if (MV_OK != nRet)
		{
			printf("Enum LightController Interfaces fail! nRet [0x%x]\n", nRet);
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

		// 创建光源控制卡句柄
		nRet = MV_CC_CreateInterface(&hInterface, stInterfaceInfoList.pInterfaceInfos[nIndex]);
		if (MV_OK == nRet)
		{
			printf("Create Interface success!\n");
		}
		else
		{
			printf("Create Interface Handle fail! nRet [0x%x]\n", nRet);
			break;
		}

		// 打开光源控制卡
		nRet = MV_CC_OpenInterface(hInterface, NULL);
		if (MV_OK == nRet)
		{
			printf("Open Interface success!\n");
		}
		else
		{
			printf("Open Interface fail! nRet [0x%x]\n", nRet);
			break;
		}

		// 打开光源控制卡事件通知并注册该事件的回调
		nRet = MV_CC_EventNotificationOn(hInterface, "CardPacketReceived0");
		if (nRet != MV_OK)
		{
			printf("EventNotificationOn fail! nRet [0x%x]\n", nRet);
			break;
		}

		nRet = MV_CC_RegisterEventCallBackEx(hInterface, "CardPacketReceived0", EventCallBack, NULL);
		if (MV_OK != nRet)
		{
			printf("Register Event CallBack fail! nRet [0x%x]\n", nRet);
			break;
		}

		// 光源控制卡常规参数的获取与修改
		Set_Get_Enum("TimerSelector");
		Set_Get_Enum("TimerTriggerSource");
		Set_Get_Bool("LineStatus");
		Set_Get_Int("TimerDuration");
		Set_Get_Int("TimerDelay");


		//关闭光源控制卡
		nRet = MV_CC_CloseInterface(hInterface);
		if (MV_OK == nRet)
		{
			printf("Close Interface success!\n");
		}
		else
		{
			printf("Close Interface Handle fail! nRet [0x%x]\n", nRet);
			break;
		}

		//销毁光源控制卡句柄
		nRet = MV_CC_DestroyInterface(hInterface);
		if (MV_OK == nRet)
		{
			printf("Destroy Interface success!\n");
		}
		else
		{
			printf("Destroy Interface Handle fail! nRet [0x%x]\n", nRet);
			break;
		}
		hInterface = NULL;
	} while (0);

	if (hInterface != NULL)
	{
		MV_CC_CloseInterface(hInterface);
		MV_CC_DestroyInterface(hInterface);
		hInterface = NULL;
	}

	// ch:反初始化SDK | en:Finalize SDK
	MV_CC_Finalize();

	printf("Press a key to exit.\n");
	PressEnterToExit();
}
