/***************************************************************************************************
 * @file      ParametrizeCamera_LoadAndSave.java
 * @breif     Use functions provided in MvCameraControlWrapper.jar to grab images
 * @author    zhanglei72
 * @date      2020/01/12
 *
 * @warning
 * @version   V1.0.0  2020/01/12 Create this file
 *            V1.0.1  2020/02/10 add parameter checking
 * @since     2020/02/10
 **************************************************************************************************/

import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class ParametrizeCamera_LoadAndSave
{
	public static Scanner scanner;
	static    Handle hCamera = null;
    static int nRet          = MV_OK;
    static boolean flag=true;
	static int camIndex = -1;
    private static void printDeviceInfo(MV_CC_DEVICE_INFO stDeviceInfo)
    {
        if (null == stDeviceInfo) {
            System.out.println("stDeviceInfo is null");
            return;
        }

		if ((stDeviceInfo.transportLayerType == MV_GIGE_DEVICE) ||( stDeviceInfo.transportLayerType == MV_GENTL_GIGE_DEVICE))
		{
            System.out.println("\tCurrentIp:       " + stDeviceInfo.gigEInfo.currentIp);
            System.out.println("\tModel:           " + stDeviceInfo.gigEInfo.modelName);
            System.out.println("\tUserDefinedName: " + stDeviceInfo.gigEInfo.userDefinedName);
        } else if (stDeviceInfo.transportLayerType == MV_USB_DEVICE) {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.usb3VInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.usb3VInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.usb3VInfo.deviceNumber);
        } else if (stDeviceInfo.transportLayerType == MV_GENTL_CAMERALINK_DEVICE)
        {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.cmlInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.cmlInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.cmlInfo.DeviceID);
        }
        else if (stDeviceInfo.transportLayerType == MV_GENTL_CXP_DEVICE)
        {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.cxpInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.cxpInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.cxpInfo.DeviceID);
        }
        else if (stDeviceInfo.transportLayerType == MV_GENTL_XOF_DEVICE)
        {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.xofInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.xofInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.xofInfo.DeviceID);
        }else {
            System.err.print("Device is not supported! \n");
        }

        System.out.println("\tAccessible:      "
            + MvCameraControl.MV_CC_IsDeviceAccessible(stDeviceInfo, MV_ACCESS_Exclusive));
        System.out.println("");
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
			String input = scanner.nextLine();
            try
            {
               
				camIndex = Integer.parseInt(input);
                if ((camIndex >= 0 && camIndex < stDeviceList.size()) || -1 == camIndex)
                {
                    break;
                }
                else
                {
                    System.out.println("Input error: " + camIndex + " Over Range:( 0 - " + (stDeviceList.size()-1) + " )");
                }
            }
            catch (NumberFormatException e)
            {
			    System.out.println("Input not number.");
                camIndex = -1;
                break;
            }
        }
       if (-1 == camIndex) {
            System.out.println("Bye.");
            return camIndex;
        }

        if (0 <= camIndex && stDeviceList.size() > camIndex)
        {
			if ((MV_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType)||(MV_GENTL_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType))
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).gigEInfo.userDefinedName);
            }
            else if (MV_USB_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).usb3VInfo.userDefinedName);
            }
			else if (MV_GENTL_CAMERALINK_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
				System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).cmlInfo.DeviceID);
            }
            else if (MV_GENTL_CXP_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
               System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).cxpInfo.DeviceID);
            }
            else if (MV_GENTL_XOF_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).xofInfo.DeviceID);
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
	


    
    public static void main(String[] args)
    {
        ArrayList<MV_CC_DEVICE_INFO> stDeviceList = null;
		scanner = new Scanner(System.in);
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
			
            // Enumerate devices
            try
            {
                stDeviceList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE);
            }
            catch (CameraControlException e)
            {
                System.err.println("Enumrate devices failed!" + e.toString());
                e.printStackTrace();
                break;
            }
			
			if (0 >= stDeviceList.size())
            {
                    System.out.println("No devices found!");
                    break;
            }
            int i = 0;
            for (MV_CC_DEVICE_INFO stDeviceInfo : stDeviceList)
            {
                if (null == stDeviceInfo)
                {
                    continue;
                }
                System.out.println("[camera " + (i++) + "]");
                printDeviceInfo(stDeviceInfo);
            }

            // choose camera
            camIndex = chooseCamera(stDeviceList);
            if (-1 == camIndex)
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
                System.err.printf("Connect to camera failed, errcode: [%#x]\n",nRet);
                break;
            }
			else
			{
				 System.err.printf("Connect suc.\n");
			}

           
         
            System.out.println("Start export the camera properties to the file\n");
            System.out.println("Wait......\n");

			
            nRet = MvCameraControl.MV_CC_FeatureSave(hCamera, "FeatureFile.mfs");
            if (MV_OK != nRet)
            {
                System.err.printf("Save Feature fail! nRet [0x%x]\n", nRet);
                break;
            }
            System.out.println("Finish export the camera properties to the file\n\n");

            System.out.println("Start import the camera properties from the file\n");
            System.out.println("Wait......\n");
        
		    // Import the camera properties from the file
            nRet =  MvCameraControl.MV_CC_FeatureLoad(hCamera, "FeatureFile.mfs");
             if (MV_OK != nRet)
            {
                System.err.printf("Feature Load failed, errcode: [%#x]\n",nRet);
                break;
            }
			else
			{
				 System.err.printf("Feature Load suc.\n");
			}
			
            System.out.println("Finish import the camera properties from the file\n");

            // Close device
            nRet = MvCameraControl.MV_CC_CloseDevice(hCamera);
            if (MV_OK != nRet)
            {
               System.err.printf("ClosDevice fail! nRet [0x%x]\n", nRet);
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
