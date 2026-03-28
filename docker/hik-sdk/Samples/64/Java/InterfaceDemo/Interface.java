/***************************************************************************************************
 * @file      Interface.java
 * @breif     Use functions provided in MvCameraControlWrapper.jar to grab images
 * @author    guohongli
 * @date      2024/02/22
 *
 * @warning
 * @version   V4.3.2.  2024/02/22 Create this file
 *            
 **************************************************************************************************/

import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class Interface
{
	public static Scanner scanner;
	static Handle hInterface = null;
    private static void printInterfaceInfo(MV_INTERFACE_INFO stInterfaceInfo)
    {
        if (null == stInterfaceInfo) {
            System.out.println("stInterfaceInfo is null");
            return;
        }

	   System.out.println("\tTLayerType: " + stInterfaceInfo.TLayerType);
       System.out.println("\tPCIEInfo: " + stInterfaceInfo.PCIEInfo);
	   System.out.println("\tSerial number: " + stInterfaceInfo.serialNumber);
	   System.out.println("\tmodel name: " + stInterfaceInfo.modelName);
	   System.out.println("\tInterfaceID: " + stInterfaceInfo.InterfaceID);
	   System.out.println("\n");
    }

	
	public static void Set_Get_Enum(String str)
	{
		
		MVCC_ENUMVALUE stEnumValue = new MVCC_ENUMVALUE();
	    MVCC_ENUMENTRY stEnumentryInfo = new MVCC_ENUMENTRY();
		int nRet = MvCameraControl.MV_CC_GetEnumValue(hInterface, str, stEnumValue);
	    if (MV_OK != nRet)
		{
			 System.err.printf("Get %s Fail! nRet [0x%x]\n", str, nRet);
			return;
		}

		stEnumentryInfo.Value = stEnumValue.curValue;
		nRet = MvCameraControl.MV_CC_GetEnumEntrySymbolic(hInterface,str, stEnumentryInfo);
		if (MV_OK != nRet)
		{
			 System.err.printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Get " + str + " = " + stEnumentryInfo.Symbolic + "Success!");
		}

		nRet = MvCameraControl.MV_CC_SetEnumValue(hInterface,str,stEnumValue.curValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Set "+ str + " =  " + stEnumentryInfo.Symbolic +" Success!");
		}
	}
	

	public static void  Set_Get_Bool(String str)
	{
		boolean bValue = false;
		int nRet = MvCameraControl.MV_CC_GetBoolValue(hInterface,str, bValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Get " + str + " = " + bValue + " Success!" );
		}

		nRet = MvCameraControl.MV_CC_SetBoolValue(hInterface,str, bValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Set" + str + " = "  + bValue + " Success");
		}
	}

	public static void Set_Get_Int(String str)
	{
		MVCC_INTVALUE stIntValue = new MVCC_INTVALUE();
		int nRet = MvCameraControl.MV_CC_GetIntValue(hInterface,str,stIntValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Get" + str + " = " + stIntValue.curValue + " Success!");
		}
	
		nRet = MvCameraControl.MV_CC_SetIntValue(hInterface,str,stIntValue.curValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Set " + str + " = " + stIntValue.curValue + " Success!");
		}
	}

	public static void Set_Get_String(String str)
	{
		MVCC_STRINGVALUE StringValue = new MVCC_STRINGVALUE();
		int nRet = MvCameraControl.MV_CC_GetStringValue(hInterface,str, StringValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Get " + str + " = "  + StringValue.curValue +     " Success!");
		}

		nRet = MvCameraControl.MV_CC_SetStringValue(hInterface,str, StringValue.curValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Set " + str + " = "  + StringValue.curValue + " Success!");
		}
	}

	public static void Set_Get_Float(String str)
	{
		MVCC_FLOATVALUE FloatValue = new MVCC_FLOATVALUE();
		int nRet = MvCameraControl.MV_CC_GetFloatValue(hInterface,str, FloatValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Get %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Get " + str + " = "  + FloatValue.curValue+ " Success!");
		}

		nRet = MvCameraControl.MV_CC_SetFloatValue(hInterface,str, FloatValue.curValue);
		if (MV_OK != nRet)
		{
			System.err.printf("Set %s Fail! nRet [0x%x]\n", str,nRet);
			return;
		}
		else
		{
			System.out.println("Set " + str + " = "  + FloatValue.curValue+ " Success!");
		}
	}

	
	
    public static int chooseEnumInterfaceType()
	{
		int EnumInterfaceTypeIndex = -1;
		
		
		while (true)
        {
			System.out.print("Please Input Enum Interfaces Type: (0 to 3):");
			if (scanner.hasNextInt()) 
			{
				try
                {
               
				    EnumInterfaceTypeIndex = scanner.nextInt();
                    if ((EnumInterfaceTypeIndex >= 0 && EnumInterfaceTypeIndex < 4) || -1 == EnumInterfaceTypeIndex)
                    {
                        break;
                    }
                    else
                    {
                       System.out.println("Input error: " + EnumInterfaceTypeIndex + " Over Range:( 0 - 3 )" );
                    }
                }
                catch (NumberFormatException e)
                {
			        System.out.println("Input not number.");
                    EnumInterfaceTypeIndex = -1;
                    break;
                }
			}
			else
			{
				EnumInterfaceTypeIndex = -1;
                break;
			}
        }
		
		
		if (-1 == EnumInterfaceTypeIndex)
        {
            System.out.println("InterfaceType index error.exit");  
        }

		return EnumInterfaceTypeIndex;
	}
	
    public static int chooseInterface(ArrayList<MV_INTERFACE_INFO> stInterfaceList)
    {
        if (null == stInterfaceList)
        {
            return -1;
        }
        
        // Choose a Interface to operate
        int InterfaceIndex = -1;
       
        while (true)
        {
			System.out.print("Please input interface index (-1 to quit):");
			if (scanner.hasNextInt()) 
			{
				try
                {
					InterfaceIndex = scanner.nextInt();
					if ((InterfaceIndex >= 0 && InterfaceIndex < stInterfaceList.size()) || -1 == InterfaceIndex)
					{
						break;
					}
					else
					{
						System.out.println("Input error: " + InterfaceIndex);
					}
				}
				catch (Exception e)
				{
					e.printStackTrace();
					InterfaceIndex = -1;
					break;
				}
			}
			else{
				InterfaceIndex = -1;
				break;
			}
           
        }
      
        if (-1 == InterfaceIndex)
        {
            System.out.println("Input error.exit");
            return InterfaceIndex;
        }

        if (0 <= InterfaceIndex && stInterfaceList.size() > InterfaceIndex)
        {
            if (MV_GIGE_INTERFACE == stInterfaceList.get(InterfaceIndex).TLayerType)
            {
                System.out.println("Connect to Interface[" + InterfaceIndex + "]: " + stInterfaceList.get(InterfaceIndex).serialNumber);
            }
            else if (MV_CAMERALINK_INTERFACE == stInterfaceList.get(InterfaceIndex).TLayerType)
            {
                System.out.println("Connect to Interface[" + InterfaceIndex + "]: " + stInterfaceList.get(InterfaceIndex).serialNumber);
            }
			else if (MV_CXP_INTERFACE == stInterfaceList.get(InterfaceIndex).TLayerType)
            {
                System.out.println("Connect to Interface[" + InterfaceIndex + "]: " + stInterfaceList.get(InterfaceIndex).serialNumber);
            }
			else if (MV_XOF_INTERFACE == stInterfaceList.get(InterfaceIndex).TLayerType)
            {
                System.out.println("Connect to Interface[" + InterfaceIndex + "]: " + stInterfaceList.get(InterfaceIndex).serialNumber);
            }
            else
            {
                System.out.println("Interface is not supported.");
            }
        }
        else
        {
            System.out.println("Invalid index " + InterfaceIndex);
            InterfaceIndex = -1;
        }

        return InterfaceIndex;
    }
    
    public static void main(String[] args)
    {
        int nRet = MV_OK;
		
        scanner = new Scanner(System.in);
		System.out.println("[0]: GIGE Interface");
		System.out.println("[1]: CAMERALINK Interface");
	    System.out.println("[2]: CXP Interface");
	    System.out.println("[3]: XOF Interface");
		

		
        do
        {
			System.out.println("SDK Version " + MvCameraControl.MV_CC_GetSDKVersion());
            
			// Initialize SDK
		    nRet = MvCameraControl.MV_CC_Initialize();
		    if (MV_OK != nRet)
		    {
			   System.err.printf("Initialize SDK fail! nRet [0x%x]\n\n",nRet);
               break;
		    }
		
            
			ArrayList<MV_INTERFACE_INFO> stInterfaceList = null;
			int nEnumInterfaceType = -1;
            try
            {
				// choose interface
			    nEnumInterfaceType = chooseEnumInterfaceType();
				if (-1 == nEnumInterfaceType)
				{
					break;
				}
				
				switch(nEnumInterfaceType)
				{
				 case 0:
				 {
					stInterfaceList = MvCameraControl.MV_CC_EnumInterfaces(MV_GIGE_INTERFACE);
					break;
				 }
				 case 1:
				 {
					stInterfaceList  = MvCameraControl.MV_CC_EnumInterfaces(MV_CAMERALINK_INTERFACE);
					break;
				 }
				 case 2:
				 {
					stInterfaceList = MvCameraControl.MV_CC_EnumInterfaces(MV_CXP_INTERFACE);
					break;
				 }
				 case 3:
				 {
					stInterfaceList = MvCameraControl.MV_CC_EnumInterfaces(MV_XOF_INTERFACE);
					break;
				 }
				 default:
				 {
					System.out.println("Input InerfaceType error!");
					break;
				 }
			   }

			   //枚举采集卡设备
			   if (0 >= stInterfaceList.size())
               {
                   System.out.println("No interface found!");
                   break;
                }
		
		
                int i = 0;
			    for (MV_INTERFACE_INFO stInterfaceInfo : stInterfaceList)
                {
                    if (null == stInterfaceInfo)
                    {
                      continue;
                    }
                    System.out.println("[Interface " + (i++) + "]");
                    printInterfaceInfo(stInterfaceInfo);
                }
            }
            catch (CameraControlException e)
            {
                System.err.println("Enumrate interface failed!" + e.toString());
                e.printStackTrace();
                break;
            }

            // choose interface
			int InterfaceIndex =-1;
            InterfaceIndex = chooseInterface(stInterfaceList);
            if (-1 == InterfaceIndex)
            {
                break;
            }

            // Create device handle
            try
            {
                hInterface = MvCameraControl.MV_CC_CreateInterface(stInterfaceList.get(InterfaceIndex));
            }
            catch (CameraControlException e)
            {
                System.err.println("Create handle failed!" + e.toString());
                e.printStackTrace();
                hInterface = null;
                break;
            }

            // Open selected device
            nRet = MvCameraControl.MV_CC_OpenInterface(hInterface,null);
            if (MV_OK != nRet)
            {
                System.err.printf("Connect to Interface failed, errcode: [%#x]\n",nRet);
                break;
            }
			else
			{
				 System.out.println("Connect suc.");
			}
			
			//采集卡属性操作,不同类型卡对各自常用属性进行获取和设置
			switch(nEnumInterfaceType)
			{
			case 0:
				{
					//MV_GIGE_INTERFACE卡属性获取和设置操作
					Set_Get_Enum("StreamSelector");
					Set_Get_Enum("TimerSelector");
					Set_Get_Enum("TimerTriggerSource");
					Set_Get_Enum("TimerTriggerActivation");
					Set_Get_Bool("HBDecompression");
					Set_Get_Int("TimerDuration");
					Set_Get_Int("TimerDelay");
			        Set_Get_Int("TimerFrequency");
					break;
				}
			case 1:
				{
					//MV_CAMERALINK_INTERFACE卡属性操作
					Set_Get_Enum("StreamSelector");
					Set_Get_Enum("CameraType");
					Set_Get_Enum("StreamPartialImageControl");
					Set_Get_Int("ImageHeight");
					Set_Get_Int("FrameTimeoutTime");
					break;
				}
			case 2:
				{
					//MV_CXP_INTERFACE卡属性操作
					Set_Get_Enum("StreamSelector");
					Set_Get_Int("StreamEnableStatus");
					Set_Get_Bool("BayerCFAEnable");
					Set_Get_Bool("GammaEnable");
					Set_Get_Float("Gamma");
					break;
				}
			case 3:
				{
					//MV_XOF_INTERFACE卡属性操作
					Set_Get_Enum("StreamSelector");
					Set_Get_Int("FrameTimeoutTime");
					Set_Get_Bool("MinFrameDelay");
					Set_Get_Enum("LinkSelector");
					break;
				}
			default:
				{
					System.err.printf("Input error!\n");
					break;
				}
			}
			
			//关闭采集卡
			nRet = MvCameraControl.MV_CC_CloseInterface(hInterface);
			if (MV_OK == nRet)
			{
				System.out.println("Close Interface success!\n");
			}
			else
			{
				System.err.printf("Close Interface Handle fail! nRet [0x%x]\n", nRet);
				break;
			}

			//销毁采集卡句柄
			nRet = MvCameraControl.MV_CC_DestroyInterface(hInterface);
			if (MV_OK == nRet)
			{
				System.out.println("Destroy Interface success!");
			}
			else
			{
				System.err.printf("Destroy Interface Handle fail! nRet [0x%x]\n", nRet);
				break;
			}
			hInterface = null;
	
        } while (false);

        if (null != hInterface)
        {
			MvCameraControl.MV_CC_CloseInterface(hInterface);
            // Destroy handle
            nRet = MvCameraControl.MV_CC_DestroyHandle(hInterface);
            if (MV_OK != nRet) 
			{
                System.err.printf("DestroyHandle failed, errcode: [%#x]\n", nRet);
            }
        }
		
		 MvCameraControl.MV_CC_Finalize();
		 scanner.close();
    }
}
