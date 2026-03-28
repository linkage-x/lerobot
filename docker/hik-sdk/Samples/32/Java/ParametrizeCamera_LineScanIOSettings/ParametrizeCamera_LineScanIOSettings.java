/***************************************************************************************************
 * @file      ParametrizeCamera_LineScanIOSettings.java
 * @breif     This sample shows how to modify FrameBurstStart and LineStart mode of the linear camera.
 * @author    *
 * @date      2024/07/18
 *
 * @warning
 * @version   V1.0.0  2024/07/18 create this file.
 * @since     2024/07/18
 **************************************************************************************************/

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class ParametrizeCamera_LineScanIOSettings
{
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

    public static void main(String[] args)
    {
        int nRet = MV_OK;
        int camIndex = -1;
        Handle hCamera = null;
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
			
            // Enumerate GigE and USB devices
            try
            {
                stDeviceList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE );
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

			System.out.print("Please Input trigger selector index(0-1): 0-FrameBurstStart, 1-LineStart\n");
            int nTriggerSelector = -1;
			while(true)
			{
				if (scanner.hasNextInt()) 
			    {
					try
					{
						nTriggerSelector = scanner.nextInt();
						if ((nTriggerSelector >= 0 && nTriggerSelector < 2))
						{
							break;
						}
						else
						{
							System.out.println("Input error: " + nTriggerSelector + " Over Range:( 0 - " + "1" + " )");
							continue;
						}
					}
					catch (Exception e)
					{
						System.out.println("Input not number.");
						nTriggerSelector = -1;
						continue;
					}
				}
				else
				{
					nTriggerSelector = -1;
					continue;
				}
			}

            if (nTriggerSelector == 0)
		    {
			    // ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerSelector", 6);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger Selector fail! nRet [0x%x]\n", nRet);
					break;
			    }

			    // ch:设置触发模式为on | en:Set trigger mode as on
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerMode", 1);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置触发源为Line0 | en:Set trigger source as Line0
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerSource", 0);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger source fail! nRet [0x%x]\n", nRet);
				    break;
			    }
		    }
		    else if (nTriggerSelector == 1)
		    {
			    // ch:设置触发选项为LineStart | en:Set trigger selector as LineStart
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerSelector", 9);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger Selector fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置触发模式为on | en:Set trigger mode as on
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerMode", 1);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置触发源为EncoderModuleOut | en:Set trigger source as EncoderModuleOut
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "TriggerSource", 6);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set Trigger source fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置编码器选项为Encoder0 | en:Set encoder selector as Encoder0
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "EncoderSelector", 0);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set encoder selector fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置编码器数据源A为Line1 | en:Set encoder source A as Line1
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "EncoderSourceA", 1);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set encoder sourceA fail! nRet [0x%x]\n", nRet);
				    break;
			    }

			    // ch:设置编码器数据源B为Line3 | en:Set encoder source B as Line3
			    nRet = MvCameraControl.MV_CC_SetEnumValue(hCamera, "EncoderSourceB", 3);
			    if (MV_OK != nRet)
			    {
                    System.err.printf("Set encoder sourceB fail! nRet [0x%x]\n", nRet);
				    break;
			    }
		    }
		    else
		    {
                System.err.printf("Input error!\n");
			    break;
		    }
		    
			// Register image callback
            nRet = MvCameraControl.MV_CC_RegisterImageCallBack(hCamera, new CameraImageCallBack() {
                @Override
                public int OnImageCallBack(byte[] bytes,MV_FRAME_OUT_INFO mv_frame_out_info)
                {
                    printFrameInfo(mv_frame_out_info);
                    return 0;
                }
            });
            if (MV_OK != nRet)
            {
                System.err.printf("register image callback failed, errcode: [%#x]\n", nRet);
                break;
            }


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
