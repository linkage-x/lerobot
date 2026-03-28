/***************************************************************************************************
 * @file      ImageSave.java
 * @breif     This demo show how to save raw/jpeg/bmp/tiff image to local
 * @author    **
 * @date      2024/07/26
 *
 **************************************************************************************************/

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.NoSuchElementException;
import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class ImageSave
{
	public static Handle hCamera = null;
	public static Scanner scanner;
   
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
			MV_CC_HB_DECODE_PARAM stDecodeParam = new MV_CC_HB_DECODE_PARAM();
            byte[] pData = new byte[(int)stParam.curValue];
			byte[] pDstBuf = new byte[(int) stParam.curValue];
			
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(hCamera, pData, stImageInfo, 1000);
            if (MV_OK != nRet)
            {
                System.err.printf("GetOneFrameTimeout fail, errcode:[%#x]\n", nRet);
                break;
            }

			System.out.println("GetOneFrame: ");
			printFrameInfo(stImageInfo);
		
			String input = "";
			
		System.out.printf("Please Input Save Image Type(raw/Jpeg/bmp/tiff/png):");
     while (true)
        {
            
            if (scanner.hasNextLine())
            {
                try
                {
                
                    input = scanner.nextLine();
                    if( (input.equalsIgnoreCase("raw")) ||(input.equalsIgnoreCase("Jpeg"))||(input.equalsIgnoreCase("bmp")) || (input.equalsIgnoreCase("tiff")) ||(input.equalsIgnoreCase("png")) )
					{
							break;
					} 
            		else if((input.equals("")) ||(input.equals(" ")))
           			 {
            			continue;
            		 }
                    else
                    {
                        System.out.printf("Input error.Please Input Save Image Type(raw/Jpeg/bmp/tiff/png)");
                    }
                }
                catch (Exception e)
                {
                    break;
                }
            }
            else
			{
                 break;
            }

        }
			
			
		
			boolean bHBFlag = IsHBPixelFormat(stImageInfo.pixelType);
			if(true == bHBFlag)
			{
				stDecodeParam.pSrcBuf = pData;
				stDecodeParam.nSrcLen = stImageInfo.frameLen;
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
			}
			
            if (input.equalsIgnoreCase("raw"))
            {
				if(true == bHBFlag)
				{
					// Save buffer content to file
					String chRawPath = String.format("Image_w%d_h%d_fn03%d.raw", stDecodeParam.nWidth, stDecodeParam.nHeight, stImageInfo.frameNum);
                    saveDataToFile(pDstBuf, stDecodeParam.nDstBufLen, chRawPath);
				}
				else
				{
					String chRawPath = String.format("Image_w%d_h%d_fn03%d.raw",  stImageInfo.ExtendWidth, stImageInfo.ExtendHeight, stImageInfo.frameNum);
                    saveDataToFile(pData, stImageInfo.frameLen, chRawPath);
				}
                    
            }
            else if(input.equalsIgnoreCase("Jpeg"))
            {
				if(true == bHBFlag)
				{
					nRet = SaveImageForHB(MV_SAVE_IAMGE_TYPE.MV_Image_Jpeg,pDstBuf,stDecodeParam);
                    if(MV_OK == nRet)
					{
						System.out.println("Save Image Jpeg success.\n");
					}
					else
					{
						System.err.printf("Save image Jpeg failed.nRet:[0x%x]\n",nRet);
					}
				}
				else
				{
					nRet = SaveImage(MV_SAVE_IAMGE_TYPE.MV_Image_Jpeg,pData,stImageInfo);
                    if(MV_OK == nRet)
					{
						System.out.println("Save Image Jpeg success.\n");
					}
					else
					{
						System.err.printf("Save image Jpeg failed.nRet:[0x%x]\n",nRet);
					}
				}      
            }
            else if(input.equalsIgnoreCase("bmp"))
            {
				if(true == bHBFlag)
				{
					nRet = SaveImageForHB(MV_SAVE_IAMGE_TYPE.MV_Image_Bmp,pDstBuf,stDecodeParam);
                    if(MV_OK == nRet)
					{
						System.out.println("Save Image bmp success.\n");
					}
					else
					{
						System.err.printf("Save image bmp failed.nRet:[0x%x]\n",nRet);
					}
				}
				else
				{
					nRet = SaveImage(MV_SAVE_IAMGE_TYPE.MV_Image_Bmp,pData,stImageInfo);
					if(MV_OK == nRet)
					{
						System.out.println("Save Image bmp success.\n");
					}
					else
					{
						System.err.printf("Save image bmp failed.nRet:[0x%x]\n",nRet);
					}
				}
            }
            else if(input.equalsIgnoreCase("tiff"))
            {
				if(true == bHBFlag)
				{
					nRet = SaveImageForHB(MV_SAVE_IAMGE_TYPE.MV_Image_Tif,pDstBuf,stDecodeParam);
                    if(MV_OK == nRet)
					{
						System.out.println("Save Image tif success.\n");
					}
					else
					{
						System.err.printf("Save image tif failed.nRet:[0x%x]\n",nRet);
					}
				}
				else
				{
					nRet = SaveImage(MV_SAVE_IAMGE_TYPE.MV_Image_Tif,pData,stImageInfo);
					if(MV_OK == nRet)
					{
						System.out.println("Save Image tif success.\n");
					}
					else
					{
						System.err.printf("Save image tif failed.nRet:[0x%x]",nRet);
					}
				}     
            }
            else if(input.equalsIgnoreCase("png"))
            {
				if(true == bHBFlag)
				{
					nRet = SaveImageForHB(MV_SAVE_IAMGE_TYPE.MV_Image_Png,pDstBuf,stDecodeParam);
                    if(MV_OK == nRet)
					{
						System.out.println("Save Image png success.\n");
					}
					else
					{
						System.err.printf("Save image png failed.nRet:[0x%x]\n",nRet);
					}
				}
				else
				{
					nRet = SaveImage(MV_SAVE_IAMGE_TYPE.MV_Image_Png,pData,stImageInfo);
					if(MV_OK == nRet)
					{
						System.out.println("Save Image png success.\n");
					}
					else
					{
						System.err.printf("Save image png failed.nRet:[0x%x]",nRet);
					}  
				}    
            }
            else
            {
                System.out.println("Input error.\n");;
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
	 private static void printDeviceInfo(MV_CC_DEVICE_INFO stDeviceInfo)
    {
        if (null == stDeviceInfo) {
            System.out.println("stDeviceInfo is null");
            return;
        }

        if ((stDeviceInfo.transportLayerType == MV_GIGE_DEVICE)||( stDeviceInfo.transportLayerType == MV_GENTL_GIGE_DEVICE))
		{
            System.out.println("\tCurrentIp:       " + stDeviceInfo.gigEInfo.currentIp);
            System.out.println("\tModel:           " + stDeviceInfo.gigEInfo.modelName);
            System.out.println("\tUserDefinedName: " + stDeviceInfo.gigEInfo.userDefinedName);
        } 
		else if (stDeviceInfo.transportLayerType == MV_USB_DEVICE) {
            System.out.println("\tUserDefinedName: " + stDeviceInfo.usb3VInfo.userDefinedName);
            System.out.println("\tSerial Number:   " + stDeviceInfo.usb3VInfo.serialNumber);
            System.out.println("\tDevice Number:   " + stDeviceInfo.usb3VInfo.deviceNumber);
        }
		else if (stDeviceInfo.transportLayerType == MV_GENTL_CAMERALINK_DEVICE)
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
		frameInfo.append("\tlostPacket[" + stFrameInfo.lostPacket + "]");
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
           

            os = new FileOutputStream(fileName);
			if(null != os)
			{
				os.write(dataToSave, 0, dataSize);
                System.out.println("save raw to local succeed.");
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
            if ((MV_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType)||(MV_GENTL_GIGE_DEVICE == stDeviceList.get(camIndex).transportLayerType))
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).gigEInfo.userDefinedName);
            }
            else if (MV_USB_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).usb3VInfo.userDefinedName);
            }
			else if (MV_CAMERALINK_DEVICE == stDeviceList.get(camIndex).transportLayerType)
            {
                System.out.println("Connect to camera[" + camIndex + "]: " + stDeviceList.get(camIndex).camLinfo.modelName);
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

    public static int SaveImage(MV_SAVE_IAMGE_TYPE enSaveImageType,byte[] dataToSave,MV_FRAME_OUT_INFO stImageInfo)
    {
        MV_SAVE_IMAGE_TO_FILE_PARAM_EX stSaveFileParam = new MV_SAVE_IMAGE_TO_FILE_PARAM_EX();

        stSaveFileParam.imageType = enSaveImageType;						// Image format to save
        stSaveFileParam.pixelType = stImageInfo.pixelType;  // Camera pixel type
        stSaveFileParam.width      = stImageInfo.ExtendWidth;         //Width
        stSaveFileParam.height     = stImageInfo.ExtendHeight;          // Height
        stSaveFileParam.dataLen    = stImageInfo.frameLen;
        stSaveFileParam.data      = dataToSave;
        stSaveFileParam.methodValue = 1;

        // jpg image nQuality range is (50-99]
        stSaveFileParam.jpgQuality = 99;
        if (MV_SAVE_IAMGE_TYPE.MV_Image_Bmp == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d_fn%d.bmp",  stImageInfo.ExtendWidth, stImageInfo.ExtendHeight, stImageInfo.frameNum);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Jpeg == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d_fn%d.jpg",  stImageInfo.ExtendWidth, stImageInfo.ExtendHeight, stImageInfo.frameNum);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Tif == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d_fn%d.tif",  stImageInfo.ExtendWidth, stImageInfo.ExtendHeight, stImageInfo.frameNum);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Png == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d_fn%d.png",  stImageInfo.ExtendWidth, stImageInfo.ExtendHeight, stImageInfo.frameNum);
         }

        int nRet = MvCameraControl.MV_CC_SaveImageToFileEx(hCamera,stSaveFileParam);
        return nRet;
    }
	
	public static int SaveImageForHB(MV_SAVE_IAMGE_TYPE enSaveImageType,byte[] dataToSave,MV_CC_HB_DECODE_PARAM stDecodeParam)
    {
        MV_SAVE_IMAGE_TO_FILE_PARAM_EX stSaveFileParam = new MV_SAVE_IMAGE_TO_FILE_PARAM_EX();

        stSaveFileParam.imageType = enSaveImageType;						// Image format to save
        stSaveFileParam.pixelType = stDecodeParam.enDstPixelType;  // Camera pixel type
        stSaveFileParam.width      = stDecodeParam.nWidth;         //Width
        stSaveFileParam.height     = stDecodeParam.nHeight;          // Height
        stSaveFileParam.dataLen    = stDecodeParam.nDstBufLen;
        stSaveFileParam.data      = dataToSave;
        stSaveFileParam.methodValue = 1;

        // jpg image nQuality range is (50-99]
        stSaveFileParam.jpgQuality = 99;
        if (MV_SAVE_IAMGE_TYPE.MV_Image_Bmp == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d.bmp",  stDecodeParam.nWidth, stDecodeParam.nHeight);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Jpeg == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d.jpg",  stDecodeParam.nWidth, stDecodeParam.nHeight);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Tif == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d.tif",  stDecodeParam.nWidth, stDecodeParam.nHeight);
        }
        else if (MV_SAVE_IAMGE_TYPE.MV_Image_Png == stSaveFileParam.imageType)
        {
            stSaveFileParam.imagePath  = String.format("Image_w%d_h%d.png",  stDecodeParam.nWidth, stDecodeParam.nHeight);
         }

        int nRet = MvCameraControl.MV_CC_SaveImageToFileEx(hCamera,stSaveFileParam);
        return nRet;
    }

}
