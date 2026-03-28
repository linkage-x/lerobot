/***************************************************************************************************
 * @file      Events_Interface.java
 * @breif     This demo show how to switch interface events
 * @author    zhanglei72
 * @date      2020/02/10
 *
 * @warning
 * @version   V1.0.0  2020/02/10 create this file.
 * @since     2020/02/10
 **************************************************************************************************/

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class Events_Interface
{
	private static Handle _ifHandle = null;
    private static Handle _hCamera = null;
    private static volatile Boolean _grabThreadExit = false;
    private static Scanner _scanner = null;
	
	
    public static void main(String[] args)
    {
      
        int interfaceType = MV_GIGE_INTERFACE | MV_CAMERALINK_INTERFACE | MV_CXP_INTERFACE | MV_XOF_INTERFACE;
        _scanner = new Scanner(System.in);

		System.out.println("SDK Version " + MvCameraControl.MV_CC_GetSDKVersion());
            
		// Initialize SDK
		MvCameraControl.MV_CC_Initialize();
		
		ArrayList<MV_INTERFACE_INFO> stInterfaceList = null;
		int nEnumInterfaceType = -1;
        try
        {
				// Enumerate interfaces(frame grabbers)
				ArrayList<MV_INTERFACE_INFO> interfaceList = MvCameraControl.MV_CC_EnumInterfaces(interfaceType);
				if (interfaceList.size() == 0)
				{
					System.out.println("no interface found!");
					return;
				}

				printInterfaceInfo(interfaceList);

				int index = chooseInterface(interfaceList);
				if (index < 0 || index > interfaceList.size())
				{
					System.out.println("index is invalid!");
					return;
				}

				//Create interface handle
				_ifHandle = MvCameraControl.MV_CC_CreateInterface(interfaceList.get(index));

				//open interface
				int nRet = MvCameraControl.MV_CC_OpenInterface(_ifHandle, null);
				if (nRet != MV_OK)
				{
					System.out.printf("MV_CC_OpenInterface failed! nRet[%#x]\n", nRet);
					return;
				}
			
				// Turn on the specific event of interface
				nRet = MvCameraControl.MV_CC_EventNotificationOn(_ifHandle,"ReceiveImageFrameStart0");
				if (nRet != MV_OK)
				{
					System.err.printf("Event Notification On failed,nRet:[%#x]", nRet);
					return;
				}
				
				// Register image callback
            nRet = MvCameraControl.MV_CC_RegisterEventCallBack(_ifHandle, "ReceiveImageFrameStart0" ,new CameraEventCallBack() {
                @Override
                public int OnEventCallBack(MV_EVENT_OUT_INFO  info)
                {
                    printEventInfo(info);
                    return 0;
                }
            });
			
			
				//Enumerate device on the frame grabber
				ArrayList<MV_CC_DEVICE_INFO> deviceList = MvCameraControl.MV_CC_EnumDevicesByInterface(_ifHandle);
				if (deviceList.size() == 0)
				{
					System.out.println("no device found!");
					return;
				}

				printDeviceInfo(deviceList);

				// choose camera
				int camIndex = chooseCamera(deviceList);
				if (-1 == camIndex)
				{
					return;
				}

				// Create device handle
				try
				{
					_hCamera = MvCameraControl.MV_CC_CreateHandle(deviceList.get(camIndex));
				}
				catch (CameraControlException e)
				{
					System.err.println("Create handle failed!" + e.toString());
					e.printStackTrace();
					return;
				}

				// Open selected device
				nRet = MvCameraControl.MV_CC_OpenDevice(_hCamera);
				if (MV_OK != nRet)
				{
					System.err.printf("Connect to camera failed, errcode: [%#x]\n",nRet);
					return;
				}
				else
				{
					System.err.printf("Connect suc.\n");
				}


				// Turn off trigger mode
				nRet = MvCameraControl.MV_CC_SetEnumValueByString(_hCamera, "TriggerMode", "Off");
				if (MV_OK != nRet)
				{
					System.err.printf("SetTriggerMode failed, errcode: [%#x]\n", nRet);
					return;
				}

				// Start grabbing
				nRet = MvCameraControl.MV_CC_StartGrabbing(_hCamera);
				if (MV_OK != nRet)
				{
					System.err.printf("StartGrabbing failed, errcode: [%#x]\n", nRet);
					return;
				}
				else
				{
					System.err.printf("StartGrabbing  suc.\n");
				}


				_grabThreadExit = false;
				Thread thread = new Thread(new Runnable(){
					@Override
					public void run(){
                    // 在线程中执行函数
                    MV_FRAME_OUT stFrameOut = new MV_FRAME_OUT();
                    while(!_grabThreadExit)
                    {
                        int nRet = MvCameraControl.MV_CC_GetImageBuffer(_hCamera, stFrameOut, 1000);
                        // ch:获取一帧图像 | en:Get one image
                        if (MV_OK == nRet)
                        {
                          /*  System.out.println("Get Image Buffer Width:" + stFrameOut.mvFrameOutInfo.ExtendWidth
                                    + " Height: "   + stFrameOut.mvFrameOutInfo.ExtendHeight
                                    + " FrameNum: " + stFrameOut.mvFrameOutInfo.frameNum);

							*/
                            nRet = MvCameraControl.MV_CC_FreeImageBuffer(_hCamera, stFrameOut);
                            if (MV_OK != nRet)
                            {
                                System.err.printf("\n Free ImageBuffer failed, errcode: [%#x]\n", nRet);
                            }
                        }
                    }
                }
            });

            thread.start();

            _scanner.useDelimiter("");
            System.out.println("Press Enter to exit.");

            while(true)
            {
                String input = _scanner.nextLine();
                if(_scanner.hasNextLine())
                {
                    break;
                }
                else
                {
                    try {
                        Thread.sleep(1 * 10);
                    }
                    catch (InterruptedException e)
                    {
                        e.printStackTrace();
                        break;
                    }
                }
            }

            _scanner.close();

            _grabThreadExit = true;
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // Stop grabbing
            nRet = MvCameraControl.MV_CC_StopGrabbing(_hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("StopGrabbing failed, errcode: [%#x]\n", nRet);
                return;
            }

            // close device
            nRet = MvCameraControl.MV_CC_CloseDevice(_hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("CloseDevice failed, errcode: [%#x]\n", nRet);
                return;
            }
        }
		catch(CameraControlException e)
        {
            System.err.println("Enumrate interface failed!" + e.toString());
            e.printStackTrace();
        }
        finally
        {
            //关闭并销毁采集卡
            if (_ifHandle != null)
            {
                MvCameraControl.MV_CC_CloseInterface(_ifHandle);
                MvCameraControl.MV_CC_DestroyInterface(_ifHandle);
                _ifHandle = null;
            }
        }
       
		MvCameraControl.MV_CC_Finalize();
    }
	
	private static void printInterfaceInfo(ArrayList<MV_INTERFACE_INFO> interfaceList)
    {
        int i = 0;
        for (MV_INTERFACE_INFO stInterfaceInfo : interfaceList)
        {
            if (null == stInterfaceInfo)
            {
                continue;
            }
            System.out.println("[Interface " + (i++) + "]");

            System.out.println("\tTLayerType: " + stInterfaceInfo.TLayerType);
            System.out.println("\tPCIEInfo: " + stInterfaceInfo.PCIEInfo);
            System.out.println("\tSerial number: " + stInterfaceInfo.serialNumber);
            System.out.println("\tmodel name: " + stInterfaceInfo.modelName);
            System.out.println("\tInterfaceID: " + stInterfaceInfo.InterfaceID);
            System.out.println("\n");
        }
    }

	
    private  static void printDeviceInfo(ArrayList<MV_CC_DEVICE_INFO> deviceInfoList)
    {
        int i = 0;
        for (MV_CC_DEVICE_INFO deviceInfo : deviceInfoList)
        {
            if (null == deviceInfo)
            {
                continue;
            }
            System.out.println("[camera " + (i++) + "]");

            if (deviceInfo.transportLayerType == MV_GIGE_DEVICE || deviceInfo.transportLayerType == MV_GENTL_GIGE_DEVICE)
            {
                System.out.println("\tCurrentIp:       " + deviceInfo.gigEInfo.currentIp);
                System.out.println("\tModel:           " + deviceInfo.gigEInfo.modelName);
                System.out.println("\tUserDefinedName: " + deviceInfo.gigEInfo.userDefinedName);
            }
            else if (deviceInfo.transportLayerType == MV_USB_DEVICE)
            {
                System.out.println("\tUserDefinedName: " + deviceInfo.usb3VInfo.userDefinedName);
                System.out.println("\tSerial Number:   " + deviceInfo.usb3VInfo.serialNumber);
                System.out.println("\tDevice Number:   " + deviceInfo.usb3VInfo.deviceNumber);
            }
            else if (deviceInfo.transportLayerType == MV_GENTL_CAMERALINK_DEVICE)
            {
                System.out.println("\tUserDefinedName: " + deviceInfo.cmlInfo.userDefinedName);
                System.out.println("\tSerial Number:   " + deviceInfo.cmlInfo.serialNumber);
                System.out.println("\tDevice Number:   " + deviceInfo.cmlInfo.DeviceID);
            }
            else if (deviceInfo.transportLayerType == MV_GENTL_CXP_DEVICE)
            {
                System.out.println("\tUserDefinedName: " + deviceInfo.cxpInfo.userDefinedName);
                System.out.println("\tSerial Number:   " + deviceInfo.cxpInfo.serialNumber);
                System.out.println("\tDevice Number:   " + deviceInfo.cxpInfo.DeviceID);
            }
            else if (deviceInfo.transportLayerType == MV_GENTL_XOF_DEVICE)
            {
                System.out.println("\tUserDefinedName: " + deviceInfo.xofInfo.userDefinedName);
                System.out.println("\tSerial Number:   " + deviceInfo.xofInfo.serialNumber);
                System.out.println("\tDevice Number:   " + deviceInfo.xofInfo.DeviceID);
            }
            else
            {
                System.out.println("\tDevice is not supported! TransferType: " + deviceInfo.transportLayerType);
            }
        }
    }




   
   public static int chooseCamera(ArrayList<MV_CC_DEVICE_INFO> stDeviceList)
    {
        if (null == stDeviceList)
        {
            return -1;
        }
        
        // Choose a device to operate
        int camIndex = -1;

        while (true)
        {
			System.out.print("Please input camera index:");
			if (_scanner.hasNextInt()) 
			{
                try
                {
				   camIndex = _scanner.nextInt();
                   if ((camIndex >= 0 && camIndex < stDeviceList.size()) || -1 == camIndex)
                   {
                       break;
                   }
                   else
                  {
                      System.out.println("Input error: " + camIndex + " Over Range:( 0 - " + (stDeviceList.size()-1) + " )");
                   }
                }
                catch (Exception e)
                {
			       System.out.println("Input not number.");
                   camIndex = -1;
                   break;
                }
			}
			else
			{
				camIndex = -1;
                break;
			}
			
        }
       if (-1 == camIndex) {
            System.out.println("Input error.exit");
            return camIndex;
        }

        if (0 <= camIndex && stDeviceList.size() > camIndex)
        {
			//deviceInfo.transportLayerType == MV_GIGE_DEVICE || deviceInfo.transportLayerType == MV_GENTL_GIGE_DEVICE
            if ((MV_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType)||(MV_GENTL_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType))
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).gigEInfo.userDefinedName);
            }
            else if (MV_USB_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).usb3VInfo.userDefinedName);
            }
			 else if (stDeviceList.get(camIndex).transportLayerType == MV_GENTL_CAMERALINK_DEVICE)
            {
				System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).cmlInfo.userDefinedName);
            }
            else if (stDeviceList.get(camIndex).transportLayerType == MV_GENTL_CXP_DEVICE)
            {
				System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).cxpInfo.userDefinedName);
            }
            else if (stDeviceList.get(camIndex).transportLayerType == MV_GENTL_XOF_DEVICE)
            {
				System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).xofInfo.userDefinedName);
            }
            else
            {
                System.out.println("Device is not supported.TransferType: " + stDeviceList.get(camIndex).transportLayerType);
            }
        }
        else
        {
            System.out.println("Invalid index " + camIndex);
            camIndex = -1;
        }

        return camIndex;
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
			if (_scanner.hasNextInt()) 
			{
				try
                {
					InterfaceIndex = _scanner.nextInt();
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
	
	public static void printEventInfo(MV_EVENT_OUT_INFO pEventInfo)
	{
		
		long nBlockId = pEventInfo.blockIdHigh;
		nBlockId = (nBlockId << 32) + pEventInfo.blockIdLow;

		long nTimestamp = pEventInfo.timestampHigh;
		nTimestamp = (nTimestamp << 32) + pEventInfo.timestampLow;

		System.err.printf("EventName[%s], EventID[%d], BlockId[%d], Timestamp[%d]\n", 
			pEventInfo.eventName, pEventInfo.eventID,nBlockId,nTimestamp);
		
	}


}
