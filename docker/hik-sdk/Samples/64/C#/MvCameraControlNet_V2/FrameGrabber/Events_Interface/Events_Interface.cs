/*
 * 这个示例演示如何接收采集卡事件。
 * This program shows how to receive frame grabber's events.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MvCameraControl;

namespace Events_Interface
{
    class Events_Interface
    {
        private const InterfaceTLayerType IFLayerType = InterfaceTLayerType.MvGigEInterface | InterfaceTLayerType.MvCameraLinkInterface | InterfaceTLayerType.MvCXPInterface
            | InterfaceTLayerType.MvXoFInterface;

        static void DeviceEventGrabedHandler(object sender, DeviceEventArgs e)
        {
            Console.WriteLine("EventName[{0}], EventID[{1}]", e.EventInfo.EventName, e.EventInfo.EventID);
        }

        static void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);
        }

        static void Main(string[] args)
        {
            int ret = MvError.MV_OK;
            IInterface ifInstance = null;
            IDevice devInstance = null;

            SDKSystem.Initialize();

            

            try
            {
                List<IInterfaceInfo> IFInfoList;
                ret = InterfaceEnumerator.EnumInterfaces(IFLayerType, out IFInfoList);
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
                Console.Write("Please input index(0-{0:d}):", IFInfoList.Count - 1);
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

                ifInstance = InterfaceFactory.CreateInterface(IFInfoList[IFIndex]);

                // ch:打开采集卡 | en:Open interface
                ret = ifInstance.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open Interface failed:{0:x8}", ret);
                    return;
                }

               //ch：开启采集卡指定事件 | en: Turn on the specific event of interface
                ret = ifInstance.EventNotificationOn("ReceiveImageFrameStart0");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("EventNotificationOn failed:{0:x8}", ret);
                    return;
                }

                // ch:注册回调函数并订阅事件 | en:Register Event callback
               ifInstance.EventGrabber.DeviceEvent += DeviceEventGrabedHandler;
               ifInstance.EventGrabber.SubscribeEvent("ReceiveImageFrameStart0");

               // ch：枚举相机 | en:Enumerate devices
                List<IDeviceInfo> devInfoList;
                ret = ifInstance.EnumDevices(out devInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("EnumDevices failed:{0:x8}", ret);
                    return;
                }

                if (devInfoList.Count == 0)
                {
                    Console.WriteLine("No device!");
                    return;
                }

                //ch: 创建第一个相机的实例 | en: Create device instance(first device)
                devInstance = DeviceFactory.CreateDevice(devInfoList[0]);

                //ch: 打开相机   | en: Open the device
                ret = devInstance.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open device failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Open device success!");

                //ch：开启取流  | en: Start grabbing
                devInstance.StreamGrabber.FrameGrabedEvent += FrameGrabedEventHandler;
                ret = devInstance.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("StartGrabbing failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Start grabbing success!");

                Console.WriteLine("Press enter to exit");
                Console.ReadLine();

                devInstance.StreamGrabber.StopGrabbing();
                Console.WriteLine("Stop grabbing!");

                devInstance.Close();
                Console.WriteLine("Close device!");

                ifInstance.Close();
                Console.WriteLine("Close interface!");
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                //ch： 释放设备资源  | en: Release the resources of device
                if (devInstance != null)
                {
                    devInstance.Dispose();
                    devInstance = null;
                }

                //ch： 释放采集卡资源  | en: Release the resources of interface
                if (ifInstance != null)
                {
                    ifInstance.Dispose();
                    ifInstance = null;
                }
                // ch: 反初始化SDK | en: Finalize SDK
                SDKSystem.Finalize();

                Console.WriteLine("Press enter to exit");
                Console.ReadKey();
            }
        }
    }
}
