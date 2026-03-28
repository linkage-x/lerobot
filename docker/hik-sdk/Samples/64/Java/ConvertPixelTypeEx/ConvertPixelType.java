/***************************************************************************************************
 * @file      ConvertPixelType.java
 * @breif     Use functions provided in MvCameraControlWrapper.jar to convert pixel type
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

public class ConvertPixelType
{
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
        }else if (stDeviceInfo.transportLayerType == MV_GENTL_CAMERALINK_DEVICE)
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

    public static void saveDataToFile(byte[] dataToSave, int dataSize, String fileName)
    {
        OutputStream os = null;

        try
        {
			if((null == dataToSave)||(dataSize <=0))
			{
				System.out.println("saveDataToFile param error.");
				return;
			}
            // create diractory
            File tempFile = new File("dat");
            if (!tempFile.exists()) 
            {
                tempFile.mkdirs();
            }

            os = new FileOutputStream(tempFile.getPath() + File.separator + fileName);
			if(null != null)
			{
		    	os.write(dataToSave, 0, dataSize);
                System.out.println("ConvertPixelType succeed.");
			}
            
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            // close file stream
            try 
            {
                if(os != null)
				{
					 os.close();
				}
            } catch (IOException e) 
            {
                e.printStackTrace();
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
        Scanner scanner = new Scanner(System.in);

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
        scanner.close();

        if (-1 == camIndex)
        {
            System.out.println("Input error.exit");
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
        int nRet = MV_OK;
        int camIndex = -1;
        Handle hCamera = null;
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
			
            // Enumerate devices
            try
            {
                stDeviceList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE);
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

            // Get payload size
            MVCC_INTVALUE stParam = new MVCC_INTVALUE();
            nRet = MvCameraControl.MV_CC_GetIntValue(hCamera, "PayloadSize", stParam);
            if (MV_OK != nRet)
            {
                System.err.printf("Get PayloadSize fail, errcode: [%#x]\n", nRet);
                break;
            }

            // Start grabbing images
            nRet = MvCameraControl.MV_CC_StartGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("Start Grabbing fail, errcode: [%#x]\n", nRet);
                break;
            }

            // Get one frame
            MV_FRAME_OUT_INFO stImageInfo = new MV_FRAME_OUT_INFO();
            byte[] pData = new byte[(int)stParam.curValue];
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(hCamera, pData, stImageInfo, 1000);
            if (MV_OK != nRet)
            {
                System.err.printf("GetOneFrameTimeout fail, errcode:[%#x]\n", nRet);
                break;
            }

            System.out.println("GetOneFrame: ");
            printFrameInfo(stImageInfo);
			
			// set interpolation algorithm type, 0-Fast 1-Equilibrium 2-Optimal 3-Optimal plus
			nRet = MvCameraControl.MV_CC_SetBayerCvtQuality(hCamera, 1);
			if (MV_OK != nRet)
			{
				System.err.printf("set Bayer convert quality fail! nRet [0x%x]\n", nRet);
				break;
			}
			
			
            int dataSizeForRGB = stImageInfo.width * stImageInfo.height * 3;    // every RGB pixel takes 3 bytes
            byte[] pDataForRGB = new byte[dataSizeForRGB];

            // Convert pixel type to RGB8_Packed
            MV_CC_PIXEL_CONVERT_PARAM_EX stConvertParam = new MV_CC_PIXEL_CONVERT_PARAM_EX();
            stConvertParam.width = stImageInfo.ExtendWidth;                                   // image width
            stConvertParam.height = stImageInfo.ExtendHeight;                                 // image height
            stConvertParam.srcData = pData;                                             // input buffer
            stConvertParam.srcDataLen = stImageInfo.frameLen;                           // input buffer size
            stConvertParam.srcPixelType = stImageInfo.pixelType;                        // input pixel format
            stConvertParam.dstPixelType = MvGvspPixelType.PixelType_Gvsp_RGB8_Packed;   // output pixel format
            stConvertParam.dstBuffer = pDataForRGB;                                     // output buffer
            stConvertParam.dstBufferSize = dataSizeForRGB;                              // output buffer size

            nRet = MvCameraControl.MV_CC_ConvertPixelTypeEx(hCamera, stConvertParam);
            if (MV_OK != nRet)
            {
                System.err.printf("Convert PixelType fail, errcode: [%#x]\n", nRet);
                break;
            }
            else
            {
                System.out.println("Convert PixelType success.");
            }
            
            // Save buffer content to file
            saveDataToFile(pDataForRGB, dataSizeForRGB, "AfterConvert_RGB.raw");

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
    }
}
