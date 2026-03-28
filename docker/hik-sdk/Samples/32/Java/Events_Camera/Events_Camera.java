/***************************************************************************************************
 * @file      Events_Camera.java
 * @breif     This demo show how to switch camera events
 * @author    
 * @date      2024/07/20
 *
 * @warning   
 **************************************************************************************************/

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class Events_Camera
{
	public static Handle hCamera = null;
	public static Scanner scanner;
    private static void printDeviceInfo(MV_CC_DEVICE_INFO stDeviceInfo)
    {
        if (null == stDeviceInfo) {
            System.out.println("stDeviceInfo is null");
            return;
        }

        if (stDeviceInfo.transportLayerType == MV_GIGE_DEVICE) {
            System.out.println("\tCurrentIp:       " + stDeviceInfo.gigEInfo.currentIp);
            System.out.println("\tModel:           " + stDeviceInfo.gigEInfo.modelName);
            System.out.println("\tUserDefinedName: " + stDeviceInfo.gigEInfo.userDefinedName);
        } else if (stDeviceInfo.transportLayerType == MV_USB_DEVICE) {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.usb3VInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.usb3VInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.usb3VInfo.deviceNumber);
        } else {
            System.err.print("Device is not supported! \n");
        }

        System.out.println("\tAccessible:      "
            + MvCameraControl.MV_CC_IsDeviceAccessible(stDeviceInfo, MV_ACCESS_Exclusive));
        System.out.println("");
    }

    private static void printFrameInfo(MV_FRAME_OUT_INFO stFrameInfo)
    {
        if (null == stFrameInfo)
        {
            System.err.println("stFrameInfo is null");
            return;
        }

        StringBuilder frameInfo = new StringBuilder("");
        frameInfo.append(("\tFrameNum[" + stFrameInfo.frameNum + "]"));
        frameInfo.append("\tWidth[" + stFrameInfo.width + "]");
        frameInfo.append("\tHeight[" + stFrameInfo.height + "]");
        frameInfo.append(String.format("\tPixelType[%#x]", stFrameInfo.pixelType.getnValue()));

        System.out.println(frameInfo.toString());
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
			if (scanner.hasNextInt()) 
			{
                try
                {
				   camIndex = scanner.nextInt();
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
           if (MV_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).gigEInfo.userDefinedName);
            }
            else if (MV_USB_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).usb3VInfo.userDefinedName);
            }
            else
            {
                System.out.println("Device is not supported.");
            }
        }
        else
        {
            System.out.println("Invalid index " + camIndex);
            camIndex = -1;
        }

        return camIndex;
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

    public static void main(String[] args)
    {
        int nRet = MV_OK;
        int camIndex = -1;
       
		scanner = new Scanner(System.in);
        ArrayList<MV_CC_DEVICE_INFO> stDeviceList = null;

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
			
            // Enumerate  devices
            try
            {
                stDeviceList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
                if (0 >= stDeviceList.size())
                {
                    System.out.println("No devices found!");
                    break;
                }
                int i = 0;
                for (MV_CC_DEVICE_INFO stDeviceInfo : stDeviceList)
                {
                    System.out.println("[camera " + (i++) + "]");
                    printDeviceInfo(stDeviceInfo);
                }
            }
            catch (CameraControlException e)
            {
                System.err.println("Enumrate devices failed!" + e.toString());
                e.printStackTrace();
                break;
            }

            // choose camera
            camIndex = chooseCamera(stDeviceList);
            if (camIndex == -1)
            {
                break;
            }
            
            // Create device handle
            try
            {
                hCamera = MvCameraControl.MV_CC_CreateHandle(stDeviceList.get(camIndex));
            }
            catch (CameraControlException e)
            {
                System.err.println("Create handle failed!" + e.toString());
                e.printStackTrace();
                hCamera = null;
                break;
            }

            // Open selected device
            nRet = MvCameraControl.MV_CC_OpenDevice(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("Connect to camera failed, errcode: [%#x]\n", nRet);
                break;
            }

            // make sure that Trigger mode is off
            nRet = MvCameraControl.MV_CC_SetEnumValueByString(hCamera, "TriggerMode", "Off");
            if (MV_OK != nRet)
            {
                System.err.printf("SetTriggerMode failed, errcode: [%#x]\n", nRet);
                break;
            }


			// ch:开启Event | en:Set Event of ExposureEnd On
			nRet = MvCameraControl.MV_CC_EventNotificationOn(hCamera, "ExposureEnd");
			if (MV_OK != nRet)
			{
				System.err.printf("Set Event Notification On fail! nRet [0x%x]\n", nRet);
				break; 
			}

			// Register image callback
            nRet = MvCameraControl.MV_CC_RegisterEventCallBack(hCamera, "ExposureEnd" ,new CameraEventCallBack() {
                @Override
                public int OnEventCallBack(MV_EVENT_OUT_INFO  info)
                {
                    printEventInfo(info);
                    return 0;
                }
            });
		
            // Start grabbing images
            nRet = MvCameraControl.MV_CC_StartGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("Start Grabbing fail, errcode: [%#x]\n", nRet);
                break;
            }
			
			scanner.useDelimiter("");
            System.out.println("Press Enter to exit.");
            while(true)
            {
				String input = scanner.nextLine();
                if(scanner.hasNextLine())
                {
                    break;
                }
				else
                {
                    try {
                        Thread.sleep(1 * 10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
						break;
                    }
                }
                
            }

            // Stop grabbing
            nRet = MvCameraControl.MV_CC_StopGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("StopGrabbing failed, errcode: [%#x]\n", nRet);
                break;
            }
			
			nRet = MvCameraControl.MV_CC_EventNotificationOff(hCamera, "ExposureEnd");
			if (MV_OK != nRet)
            {
                System.err.printf("Event Notification Off failed, errcode: [%#x]\n", nRet);
                break;
            }
			
			// close device
            nRet = MvCameraControl.MV_CC_CloseDevice(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("CloseDevice failed, errcode: [%#x]\n", nRet);
                break;
            }
			
			
        } while (false);

        if (null != hCamera)
        {
            // Destroy handle
            nRet = MvCameraControl.MV_CC_DestroyHandle(hCamera);
            if (MV_OK != nRet) {
                System.err.printf("DestroyHandle failed, errcode: [%#x]\n", nRet);
            }
        }
		MvCameraControl.MV_CC_Finalize();
		scanner.close();
    }
}
