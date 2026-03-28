/***************************************************************************************************
 * @file      SaveImage.java
 * @breif     Use functions provided in MvCameraControlWrapper.jar to decode the lossless compressed data
 * @author    hulongcheng
 * @date      2021/10/15
 *
 * @warning
 * @version   V1.0.0  2021/10/15 Create this file
 * @since     2021/10/15
 **************************************************************************************************/

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControl.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;
public class HighBandwidthDecode {
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
        }
		else {
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
        frameInfo.append("\tframeLen[" + stFrameInfo.frameLen + "]");
        frameInfo.append(String.format("\tPixelType[%#x]", stFrameInfo.pixelType.getnValue()));

        System.out.println(frameInfo.toString());
    }

    public static void saveDataToFile(byte[] dataToSave, int dataSize, String fileName)
    {
        OutputStream os = null;

        try
        {
			if((null == dataToSave)||(dataSize <= 0))
			{
				System.out.println("saveDataToFile param error.");
				return;
			}
            // Create directory
            File tempFile = new File("dat");
            if (!tempFile.exists())
            {
                tempFile.mkdirs();
            }

            os = new FileOutputStream(tempFile.getPath() + File.separator + fileName);
			if(null != os)
			{
			   os.write(dataToSave, 0, dataSize);
               System.out.println("SaveImage succeed.");
			}
			else
			{
				System.err.printf("SaveImage fail!\n");
			}
           
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            // Close file stream
            try
            {
               if(os != null)
				{
					 os.close();
				}
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }
	
	private static boolean IsHBPixelFormat(MvGvspPixelType ePixelType)
    {
		switch (ePixelType)
		{
			case PixelType_Gvsp_HB_Mono8:
			case PixelType_Gvsp_HB_Mono10:
			case PixelType_Gvsp_HB_Mono10_Packed:
			case PixelType_Gvsp_HB_Mono12:
			case PixelType_Gvsp_HB_Mono12_Packed:
			case PixelType_Gvsp_HB_Mono16:
			case PixelType_Gvsp_HB_RGB8_Packed:
			case PixelType_Gvsp_HB_BGR8_Packed:
			case PixelType_Gvsp_HB_RGBA8_Packed:
			case PixelType_Gvsp_HB_BGRA8_Packed:
			case PixelType_Gvsp_HB_RGB16_Packed:
			case PixelType_Gvsp_HB_BGR16_Packed:
			case PixelType_Gvsp_HB_RGBA16_Packed:
			case PixelType_Gvsp_HB_BGRA16_Packed:
			case PixelType_Gvsp_HB_YUV422_Packed:
			case PixelType_Gvsp_HB_YUV422_YUYV_Packed:
			case PixelType_Gvsp_HB_BayerGR8:
			case PixelType_Gvsp_HB_BayerRG8:
			case PixelType_Gvsp_HB_BayerGB8:
			case PixelType_Gvsp_HB_BayerBG8:
			case PixelType_Gvsp_HB_BayerRBGG8:
			case PixelType_Gvsp_HB_BayerGB10:
			case PixelType_Gvsp_HB_BayerGB10_Packed:
			case PixelType_Gvsp_HB_BayerBG10:
			case PixelType_Gvsp_HB_BayerBG10_Packed:
			case PixelType_Gvsp_HB_BayerRG10:
			case PixelType_Gvsp_HB_BayerRG10_Packed:
			case PixelType_Gvsp_HB_BayerGR10:
			case PixelType_Gvsp_HB_BayerGR10_Packed:
			case PixelType_Gvsp_HB_BayerGB12:
			case PixelType_Gvsp_HB_BayerGB12_Packed:
			case PixelType_Gvsp_HB_BayerBG12:
			case PixelType_Gvsp_HB_BayerBG12_Packed:
			case PixelType_Gvsp_HB_BayerRG12:
			case PixelType_Gvsp_HB_BayerRG12_Packed:
			case PixelType_Gvsp_HB_BayerGR12:
			case PixelType_Gvsp_HB_BayerGR12_Packed:
				return true;
			default:
				return false;
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
            if (MV_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType)
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
        ArrayList<MV_CC_DEVICE_INFO> stDeviceList;

        do
        {
            System.out.println("SDK Version " + MvCameraControl.MV_CC_GetSDKVersion());

			nRet = MvCameraControl.MV_CC_Initialize();
			if (MV_OK != nRet)
		    {
			   System.err.printf("Initialize SDK fail! nRet [0x%x]\n\n",nRet);
               break;
		    }
			
            // Enuerate  devices
            try
            {
                stDeviceList = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE);
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

            // Create handle
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

            // Open device
            nRet = MvCameraControl.MV_CC_OpenDevice(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("Connect to camera failed, errcode: [%#x]\n", nRet);
                break;
            }
			
			// en:Detection network optimal package size(It only works for the GigE camera)
            if (stDeviceList.get(camIndex).transportLayerType == MV_GIGE_DEVICE)
            {
               int nPacketSize = MvCameraControl.MV_CC_GetOptimalPacketSize(hCamera);
               if (nPacketSize > 0)
                {
                  nRet = MvCameraControl.MV_CC_SetIntValue(hCamera,"GevSCPSPacketSize",nPacketSize);
                  if(nRet != MV_OK)
                  {
                     System.err.printf("Warning: Set Packet Size fail nRet [0x%x]!", nRet);
                  }
                }
                else
                {
                    System.err.printf("Warning: Get Packet Size fail nRet [0x%x]!", nPacketSize);
                }
            }

            // Make sure that trigger mode is off
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

            // Start grabbing
            nRet = MvCameraControl.MV_CC_StartGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("Start Grabbing fail, errcode: [%#x]\n", nRet);
                break;
            }
            System.err.printf("PayloadSize:[%d]\n", stParam.curValue);
            // Get one frame
            MV_FRAME_OUT_INFO stImageInfo = new MV_FRAME_OUT_INFO();
            byte[] pData = new byte[(int)stParam.curValue];

            int nImageNum = 10;
            for (int i = 0; i < nImageNum; i++) {
                nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(hCamera, pData, stImageInfo, 1000);
                if (MV_OK != nRet)
                {
                    System.err.printf("GetOneFrameTimeout fail, errcode:[%#x]\n", nRet);
                    break;
                }

                if (stImageInfo.lostPacket > 0) {
                    System.err.printf("Lost Pacekt Number %d\n", stImageInfo.lostPacket);
                    continue;
                }

                System.out.println("GetOneFrame: ");
                printFrameInfo(stImageInfo);
				
				 // 保存裸图  如果是HB 进行解码
				boolean bHBFlag = IsHBPixelFormat(stImageInfo.pixelType);
				if(true == bHBFlag)
				{
					MV_CC_HB_DECODE_PARAM stDecodeParam = new MV_CC_HB_DECODE_PARAM();
					stDecodeParam.pSrcBuf = pData;
					stDecodeParam.nSrcLen = stImageInfo.frameLen;

					byte[] pDstBuf = new byte[(int) stParam.curValue];
					stDecodeParam.pDstBuf = pDstBuf;
					stDecodeParam.nDstBufSize = pDstBuf.length;
					nRet = MvCameraControl.MV_CC_HB_Decode(hCamera, stDecodeParam);
					if (MV_OK != nRet)
					{
						System.err.printf("MV_CC_HB_Decode fail, errcode:[%#x]\n", nRet);
						break;
					}
					else
					{
						System.out.println("MV_CC_HB_Decode success.");
					}
					StringBuffer fileName = new StringBuffer();
					fileName.append("Image_");
					fileName.append(stDecodeParam.nWidth);
					fileName.append("_");
					fileName.append(stDecodeParam.nHeight);
					fileName.append("_");
					fileName.append(stImageInfo.frameNum);
					fileName.append(".raw");
					// Save buffer content to file
					saveDataToFile(pDstBuf, stDecodeParam.nDstBufLen, fileName.toString());
				}
				else
				{
					System.out.println("It is not HB compress pixelType.");
				}
            }

            // Stop grabbing
            nRet = MvCameraControl.MV_CC_StopGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("StopGrabbing fail, errcode: [%#x]\n", nRet);
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
