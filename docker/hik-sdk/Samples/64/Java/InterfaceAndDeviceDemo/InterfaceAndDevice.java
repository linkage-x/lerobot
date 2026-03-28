/*
 * 这个示例演示了先枚举采集卡，打开采集，然后通过采集卡实例枚举相机、从相机取流。
 * This sample shows how to enumerate devices on a specified frame grabber, and grabbing images from the device.
 * 
 */
import java.util.ArrayList;
import java.util.Scanner;

import MvCameraControlWrapper.*;
import static MvCameraControlWrapper.MvCameraControlDefines.*;

public class InterfaceAndDevice
{
    
    private static Handle _ifHandle = null;
    private static Handle _hCamera = null;
    private static volatile Boolean _grabThreadExit = false;
    private static Scanner _scanner = null;

    public static void main(String[] args)
    {
        // Initialize SDK
        MvCameraControl.MV_CC_Initialize();

        System.out.println("SDK Version " + MvCameraControl.MV_CC_GetSDKVersion());

        int interfaceType = MV_GIGE_INTERFACE | MV_CAMERALINK_INTERFACE | MV_CXP_INTERFACE | MV_XOF_INTERFACE;

        _scanner = new Scanner(System.in);

        try
        {
            //Enumerate interfaces(frame grabbers)
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
            int ret = MvCameraControl.MV_CC_OpenInterface(_ifHandle, null);
            if (ret != MV_OK)
            {
                System.out.printf("MV_CC_OpenInterface failed! ret[%#x]\n", ret);
                return;
            }

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
            ret = MvCameraControl.MV_CC_OpenDevice(_hCamera);
            if (MV_OK != ret)
            {
                System.err.printf("Connect to camera failed, errcode: [%#x]\n",ret);
                return;
            }
            else
            {
                System.err.printf("Connect suc.\n");
            }


            // Turn off trigger mode
            ret = MvCameraControl.MV_CC_SetEnumValueByString(_hCamera, "TriggerMode", "Off");
            if (MV_OK != ret)
            {
                System.err.printf("SetTriggerMode failed, errcode: [%#x]\n", ret);
                return;
            }

            // Start grabbing
            ret = MvCameraControl.MV_CC_StartGrabbing(_hCamera);
            if (MV_OK != ret)
            {
                System.err.printf("StartGrabbing failed, errcode: [%#x]\n", ret);
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
                        int ret = MvCameraControl.MV_CC_GetImageBuffer(_hCamera, stFrameOut, 1000);
                        // ch:获取一帧图像 | en:Get one image
                        if (MV_OK == ret)
                        {
                            System.out.println("Get Image Buffer Width:" + stFrameOut.mvFrameOutInfo.ExtendWidth
                                    + " Height: "   + stFrameOut.mvFrameOutInfo.ExtendHeight
                                    + " FrameNum: " + stFrameOut.mvFrameOutInfo.frameNum);


                            ret = MvCameraControl.MV_CC_FreeImageBuffer(_hCamera, stFrameOut);
                            if (MV_OK != ret)
                            {
                                System.err.printf("\n Free ImageBuffer failed, errcode: [%#x]\n", ret);
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
            ret = MvCameraControl.MV_CC_StopGrabbing(_hCamera);
            if (MV_OK != ret)
            {
                System.err.printf("StopGrabbing failed, errcode: [%#x]\n", ret);
                return;
            }

            // close device
            ret = MvCameraControl.MV_CC_CloseDevice(_hCamera);
            if (MV_OK != ret)
            {
                System.err.printf("CloseDevice failed, errcode: [%#x]\n", ret);
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

        //UnInitialize SDK
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
                System.err.print("Device is not supported! \n");
            }
        }
    }

    private static int chooseInterface(ArrayList<MV_INTERFACE_INFO> stInterfaceList)
    {
        // Choose a Interface to operate
        int InterfaceIndex = -1;

        while (true)
        {
            System.out.printf("Please input interface index[0-%d], -1 to exit:", stInterfaceList.size() - 1);
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
                catch (NumberFormatException e)
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
}
