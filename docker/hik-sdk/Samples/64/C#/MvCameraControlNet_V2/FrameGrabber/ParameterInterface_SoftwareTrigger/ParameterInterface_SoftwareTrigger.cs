/*
 * 这个示例演示了如何使用采集卡软触发和快速软触发功能
 * This program shows how to use the software trigger and quick software trigger of the frame grabbers. 
 * [注] 快速软触发功能需要采集卡固件支持，目前只有部分XoF(GS1104F)和CXP(GX1002,GX1004)采集卡支持
 * [PS] The quick soft trigger function requires firmware support from the frame grabber,
 *      and currently only some XoF (GS1104F) and CXP (GX1002, GX1004) frame grabbers support it.
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;
using System.Threading;

namespace ParameterInterface_SoftwareTrigger
{
    class ParameterInterface_SoftwareTrigger
    {
        static volatile bool _grabThreadExit = false;
        static volatile int _triggerIndex = -1;
        static IInterface _ifInstance = null;

        static void FrameGrabThread(object obj)
        {
            IStreamGrabber streamGrabber = (IStreamGrabber)obj;

            while(!_grabThreadExit)
            {
                IFrameOut frame;
                int ret;

                if (0 == _triggerIndex)
                {
                    // ch:通过采集卡软触发一次 | en:Software trigger once from interface
                    ret = _ifInstance.Parameters.SetCommandValue("StreamSoftwareTrigger");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Software Trigger once failed:{0:x8}", ret);
                    }
                    else
                    {
                        Console.WriteLine("Software Trigger once success");
                    }
                }
                else
                {
                    // ch:通过采集卡快速软触发一次 | en:Quick software trigger once from interface
                    ret = _ifInstance.Parameters.SetCommandValue("QuickSoftwareTrigger0");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Quick Software Trigger once failed:{0:x8}", ret);
                    }
                    else
                    {
                        Console.WriteLine("Quick Software Trigger once success");
                    }
                }

                //ch：获取一帧图像 | en: Get one frame
                ret = streamGrabber.GetImageBuffer(1000, out frame);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    continue;
                }

                Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                //Do some thing
                

                //ch: 释放图像缓存  | en: Release the image buffer
                streamGrabber.FreeImageBuffer(frame);
            }
        }

        void Run()
        {
            IDevice device = null;
            int ret = 0;
            try
            {
                Console.WriteLine("Enumerate interfaces....");
                List<IInterfaceInfo> IFInfoList;
                ret = InterfaceEnumerator.EnumInterfaces(InterfaceTLayerType.MvCameraLinkInterface | InterfaceTLayerType.MvCXPInterface
                    | InterfaceTLayerType.MvXoFInterface, out IFInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Enum interface failed:{0:x8}", ret);
                    return;
                }

                if (0 == IFInfoList.Count)
                {
                    Console.WriteLine("No interface found");
                    return;
                }

                // ch:显示采集卡信息 | en:Show interface info
                int IFIndex = 0;
                foreach (var ifInfo in IFInfoList)
                {
                    Console.WriteLine("[Interface {0}]: ", IFIndex);
                    Console.WriteLine("TLayerType: " + ifInfo.TLayerType.ToString());
                    Console.WriteLine("DisplayName: " + ifInfo.DisplayName);
                    Console.WriteLine("InterfaceID: " + ifInfo.InterfaceID);
                    Console.WriteLine("SerialNumber: " + ifInfo.SerialNumber);
                    IFIndex++;
                }

                // ch:选择采集卡 | en:Select interface
                Console.Write("Please input index(0-{0:d}) to select a interface:", IFInfoList.Count - 1);
                try
                {
                    IFIndex = Convert.ToInt32(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid Index!");
                    return;
                }

                if (IFIndex < 0 || IFIndex >= IFInfoList.Count)
                {
                    Console.WriteLine("Error Index!");
                    return;
                }

                _ifInstance = InterfaceFactory.CreateInterface(IFInfoList[IFIndex]);

                // ch:打开采集卡 | en:Open interface
                ret = _ifInstance.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open Interface failed:{0:x8}", ret);
                    return;
                }

                ret = _ifInstance.Parameters.SetBoolValue("StreamTriggerEnable", true);

                Console.Write("Use Software Trigger or Quick Software Trigger, 0.Software Trigger, 1.Quick Software Trigger:");

                try
                {
                    _triggerIndex = Convert.ToInt32(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid Index!");
                    return;
                }

                if (_triggerIndex < 0 || _triggerIndex >= 2)
                {
                    Console.WriteLine("Error Index!");
                    return;
                }

                if (0 == _triggerIndex)
                {
                    //ch: 配置Stream触发源为SoftwareSignal0 | en:Set the Stream Trigger Source to SoftwareSignal0
                    ret = _ifInstance.Parameters.SetEnumValueByString("StreamTriggerSource", "SoftwareSignal0");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set StreamTriggerSource failed:{0:x8}", ret);
                        return;
                    }
                    else
                    {
                        Console.WriteLine("Set StreamTriggerSource to SoftwareSignal0");
                    }
                }
                else
                {
                    //ch: 配置Stream触发源为QuickSoftwareSignal0 | en:Set the Stream Trigger Source to QuickSoftwareSignal0
                    ret = _ifInstance.Parameters.SetEnumValueByString("StreamTriggerSource", "QuickSoftwareTrigger0");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set StreamTriggerSource failed:{0:x8}, maybe firmware not support Quick Software Trigger", ret);
                        return;
                    }
                    else
                    {
                        Console.WriteLine("Set StreamTriggerSource to QuickSoftwareSignal0");
                    }
                }

                //ch: 配置StreamTriggerActivation为RisingEdge | en:Set the Stream Trigger Activation to RisingEdge
                ret = _ifInstance.Parameters.SetEnumValueByString("StreamTriggerActivation", "RisingEdge");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set StreamTriggerActivation failed:{0:x8}", ret);
                    return;
                }
                else
                {
                    Console.WriteLine("Set StreamTriggerActivation to RisingEdge");
                }
                

                Console.WriteLine();
                Console.WriteLine("Enumerate devices....");
                // ch:枚举采集卡上的相机 | en:Enum camera of interface
                List<IDeviceInfo> devInfoList;
                ret = _ifInstance.EnumDevices(out devInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Enum device failed:{0:x8}", ret);
                    return;
                }

                int devIndex = 0;
                foreach (var devInfo in devInfoList)
                {
                    Console.WriteLine("[Device {0}]:", devIndex);
                    if (devInfo.TLayerType == DeviceTLayerType.MvGigEDevice || devInfo.TLayerType == DeviceTLayerType.MvVirGigEDevice || devInfo.TLayerType == DeviceTLayerType.MvGenTLGigEDevice)
                    {
                        IGigEDeviceInfo gigeDevInfo = devInfo as IGigEDeviceInfo;
                        uint nIp1 = ((gigeDevInfo.CurrentIp & 0xff000000) >> 24);
                        uint nIp2 = ((gigeDevInfo.CurrentIp & 0x00ff0000) >> 16);
                        uint nIp3 = ((gigeDevInfo.CurrentIp & 0x0000ff00) >> 8);
                        uint nIp4 = (gigeDevInfo.CurrentIp & 0x000000ff);
                        Console.WriteLine("DevIP: {0}.{1}.{2}.{3}", nIp1, nIp2, nIp3, nIp4);
                    }

                    Console.WriteLine("ModelName:" + devInfo.ModelName);
                    Console.WriteLine("SerialNumber:" + devInfo.SerialNumber);
                    Console.WriteLine();
                    devIndex++;
                }

                Console.Write("Please input index(0-{0:d}) to select a device:", devInfoList.Count - 1);

                devIndex = Convert.ToInt32(Console.ReadLine());

                if (devIndex > devInfoList.Count - 1 || devIndex < 0)
                {
                    Console.Write("Input Error!\n");
                    return;
                }

                // ch:创建设备 | en:Create device
                device = DeviceFactory.CreateDevice(devInfoList[devIndex]);

                ret = device.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open device failed:{0:x8}", ret);
                    return;
                }

                //ch:设置相机为LineScan模式 | en: Set Camera Scan Mode to LineScan
                ret = device.Parameters.SetEnumValueByString("ScanMode", "LineScan");

                // ch:关闭设备触发模式 | en:Close device trigger mode
                ret = device.Parameters.SetEnumValueByString("TriggerMode", "Off");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Turn off device trigger mode failed:{0:x8}", ret);
                    return;
                }
                else
                {
                    Console.WriteLine("Set TriggerMode to Off");
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                device.StreamGrabber.SetImageNodeNum(5);

                // ch:开启抓图 || en: start grab image
                ret = device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                // ch:开启抓图线程 | en: Start the grabbing thread
                Thread GrabThread = new Thread(FrameGrabThread);
                GrabThread.Start(device.StreamGrabber);

                Console.WriteLine("Press enter to stop grabbing");
                Console.ReadLine();

                //ch: 通知线程退出 | en: Notify the grab thread to exit
                _grabThreadExit = true;
                GrabThread.Join();

                // ch:停止抓图 | en:Stop grabbing
                ret = device.StreamGrabber.StopGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Stop grabbing");

                // ch:关闭设备 | en:Close device
                ret = device.Close();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Close device failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Close device");

                // ch:关闭采集卡 | en:Close interface
                _ifInstance.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                //ch: 关闭相机，释放相机资源 | en: Close and release the resources of device
                if (device != null)
                {
                    _ifInstance.Close();
                    _ifInstance.Dispose();
                }

                //ch： 释放采集卡资源  | en: Release the resources of interface
                if (_ifInstance != null)
                {
                    _ifInstance.Close();
                    _ifInstance.Dispose();
                }
            }
        }
        static void Main(string[] args)
        {
            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

            ParameterInterface_SoftwareTrigger program = new ParameterInterface_SoftwareTrigger();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
