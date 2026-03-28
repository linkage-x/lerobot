/***************************************************************************************************
 * @file      ConnectSpecCamera.java
 * @breif     Use functions provided in MvCameraControlWrapper.jar to grab images
 * @author    hulongcheng
 * @date      2021/07/12
 *
 * @warning
 * @version   V1.0.0  2021/07/12 Create this file
 * @since     2021/07/14
 **************************************************************************************************/


import MvCameraControlWrapper.CameraControlException;
import MvCameraControlWrapper.MvCameraControl;
import MvCameraControlWrapper.MvCameraControlDefines;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

import static MvCameraControlWrapper.MvCameraControl.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class ConnectSpecCamera {

 private static void printFrameInfo(MV_FRAME_OUT_INFO stFrameInfo) {
        if (null == stFrameInfo) {
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
    static class GetOneFrameThread extends Thread {

        private Handle mHandle;
        private boolean flag = true;

        public GetOneFrameThread(Handle handle) {
            this.mHandle = handle;
        }

        public void stopThread() {
            flag = false;
        }

        @Override
        public void run() {
            super.run();

            // Get payload size
            MVCC_INTVALUE stParam = new MVCC_INTVALUE();
            int nRet = MvCameraControl.MV_CC_GetIntValue(mHandle, "PayloadSize", stParam);
            if (MV_OK != nRet) {
                System.err.printf("Get PayloadSize fail, errcode: [%#x]\n", nRet);
                return;
            }

            while (flag) {
                MV_FRAME_OUT_INFO stImageInfo = new MV_FRAME_OUT_INFO();
                byte[] pData = new byte[(int)stParam.curValue];
                nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(mHandle, pData, stImageInfo, 1000);
                if (MV_OK != nRet) {
                    System.err.printf("GetOneFrameTimeout fail, errcode:[%#x]\n", nRet);
                    break;
                } else {
                    printFrameInfo(stImageInfo);
                }
            }
        }
    }
	
	 public static boolean isNumeric(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            int d = Integer.parseInt(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    public static void main(String[] args) {
        int nRet = MV_OK;
        int camIndex = -1;
        Handle hCamera = null;
      
        Scanner scanner = new Scanner(System.in);

		MV_CC_DEVICE_INFO stDevInfo = new MV_CC_DEVICE_INFO();
	
		// The camera IP that needs to be connected (based on actual padding)
		System.out.print("Please input Current Camera Ip :");
		String strCurrentIp = scanner.nextLine(); 
		
	
		
		 int nIp1, nIp2, nIp3, nIp4, nIp;

	    String[] parts = strCurrentIp.split("\\."); 
 
        
        if (parts.length == 4 && isNumeric(parts[0]) && isNumeric(parts[1]) && isNumeric(parts[2]) &&(isNumeric(parts[3]))) {
             nIp1 = Integer.parseInt(parts[0]);
             nIp2 = Integer.parseInt(parts[1]);
			 nIp3 = Integer.parseInt(parts[2]);
			 nIp4 = Integer.parseInt(parts[3]);
        } else {
            System.out.println("Input currentIP Invalid");
             return;
        }

	
   	// The pc IP that needs to be connected (based on actual padding)
		System.out.print("Please input Net Export Ip : ");
		String strNetExport = scanner.nextLine(); 
		String[] Netparts = strNetExport.split("\\.");
		if (Netparts.length == 4 && isNumeric(Netparts[0]) && isNumeric(Netparts[1]) && isNumeric(Netparts[2]) &&(isNumeric(Netparts[3]))) {
             nIp1 = Integer.parseInt(Netparts[0]);
             nIp2 = Integer.parseInt(Netparts[1]);
			 nIp3 = Integer.parseInt(Netparts[2]);
			 nIp4 = Integer.parseInt(Netparts[3]);
        } else {
            System.out.println("Input NetIP Invalid");
            return;
        }

	
		stDevInfo.transportLayerType = MV_GIGE_DEVICE;// ch:仅支持GigE相机 | en:Only support GigE camera
		stDevInfo.gigEInfo.currentIp  = strCurrentIp;
		stDevInfo.gigEInfo.netExport = strNetExport;
	
	
        do {
            System.out.println("SDK Version " + MvCameraControl.MV_CC_GetSDKVersion());

			// Initialize SDK
		    nRet = MvCameraControl.MV_CC_Initialize();
		    if (MV_OK != nRet)
		    {
			   System.err.printf("Initialize SDK fail! nRet [0x%x]\n\n",nRet);
               break;
		    }


            // Create handle
            try {
                hCamera = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            }
            catch (CameraControlException e) {
                System.err.println("Create handle failed!" + e.toString());
                e.printStackTrace();
                hCamera = null;
                break;
            }

            // Open device
            nRet = MvCameraControl.MV_CC_OpenDevice(hCamera);
            if (MV_OK != nRet) {
                System.err.printf("Connect to camera failed, errcode: [%#x]\n", nRet);
                break;
            }
			
			

            // Make sure that trigger mode is off
            nRet = MvCameraControl.MV_CC_SetEnumValueByString(hCamera, "TriggerMode", "Off");
            if (MV_OK != nRet) {
                System.err.printf("SetTriggerMode failed, errcode: [%#x]\n", nRet);
                break;
            }

            // Start grabbing
            nRet = MvCameraControl.MV_CC_StartGrabbing(hCamera);
            if (MV_OK != nRet) {
                System.err.printf("Start Grabbing fail, errcode: [%#x]\n", nRet);
                break;
            }

            // Get frame
            GetOneFrameThread getOneFrameThread = new GetOneFrameThread(hCamera);
            getOneFrameThread.start();

            // Press enter and enter any characters to end the get frame thread
			String input = scanner.nextLine();
            while (scanner.hasNextLine()) {
                getOneFrameThread.stopThread();
                try {
                    getOneFrameThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                getOneFrameThread = null;
                break;
            }
            // Stop grabbing
            nRet = MvCameraControl.MV_CC_StopGrabbing(hCamera);
            if (MV_OK != nRet)
            {
                System.err.printf("StopGrabbing fail, errcode: [%#x]\n", nRet);
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
