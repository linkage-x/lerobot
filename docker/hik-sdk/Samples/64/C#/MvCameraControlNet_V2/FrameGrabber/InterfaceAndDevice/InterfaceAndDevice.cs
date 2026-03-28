/*
 * 这个示例演示了先枚举采集卡，打开采集，然后通过采集卡实例枚举相机、从相机取流。
 * This sample shows how to enumerate devices on a specified frame grabber, and grabbing images from the device.
 * 
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;

namespace InterfaceAndDevice
{
    class InterfaceAndDevice
    {
        /// <summary>
        /// ch:传输层类型 | en: Transport layer type
        /// </summary>
        private const InterfaceTLayerType IFLayerType = InterfaceTLayerType.MvGigEInterface | InterfaceTLayerType.MvCameraLinkInterface | InterfaceTLayerType.MvCXPInterface
            | InterfaceTLayerType.MvXoFInterface;

        static void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);
        }

        public void Run()
        {
            IInterface ifInstance = null;
            IDevice devInstance = null;
             try
            {
                 // ch: 枚举采集卡 | en: Enumerate interfaces(frame grabber)
                List<IInterfaceInfo> IFInfoList;
                Int32 ret = InterfaceEnumerator.EnumInterfaces(IFLayerType, out IFInfoList);
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
                PrintInterfaceInfo(IFInfoList);

                // ch:选择采集卡 | en:Select interface
                Console.Write("Please input index(0-{0:d}):", IFInfoList.Count - 1);

                Int32 ifIndex = Convert.ToInt32(Console.ReadLine());
                if (ifIndex < 0 || ifIndex >= IFInfoList.Count)
                {
                    Console.WriteLine("Error Index!");
                    return;
                }

                ifInstance = InterfaceFactory.CreateInterface(IFInfoList[ifIndex]);

                // ch:打开采集卡 | en:Open interface
                ret = ifInstance.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open Interface failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Open Interface success");

                // ch: 枚举采集卡上的相机 | en：Enumerate devices of interface
                List<IDeviceInfo> devInfoList;
                ret = ifInstance.EnumDevices(out devInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("EnumDevices failed:{0:x8}", ret);
                    return;
                }

                PrintDeviceInfo(devInfoList);

                Console.Write("Please input device index(0-{0:d}):", devInfoList.Count - 1);

                Int32 devIndex = Convert.ToInt32(Console.ReadLine());
                if (devIndex > devInfoList.Count - 1 || devIndex < 0)
                {
                    Console.Write("Input Error!\n");
                    return;
                }

                // ch:创建设备 | en:Create device
                devInstance = DeviceFactory.CreateDevice(devInfoList[devIndex]);

                // ch:打开设备 | en:Open device
                ret = devInstance.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open device failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Open device success");

                // ch:设置触发模式为off || en:set trigger mode as off
                ret = devInstance.Parameters.SetEnumValue("TriggerMode", 0);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                devInstance.StreamGrabber.SetImageNodeNum(5);

                // ch:注册回调函数 | en:Register image callback
                devInstance.StreamGrabber.FrameGrabedEvent += FrameGrabedEventHandler;
                // ch:开启抓图 | en: start grab image
                ret = devInstance.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Start grabbing success");

                Console.WriteLine("Press enter to stop grabbing");
                Console.ReadLine();


                // ch:停止抓图 | en:Stop grabbing
                devInstance.StreamGrabber.StopGrabbing();
                Console.WriteLine("Stop grabbing success");

                //ch: 关闭相机 | en: Close device
                devInstance.Close();
                Console.WriteLine("Close device success");

                //ch：关闭采集卡 | en: Close interface
                ifInstance.Close();
                Console.WriteLine("Close inerface success");
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                //ch：释放相机资源 | en：Release the resources of device
                if (devInstance != null)
                {
                    devInstance.Dispose();
                }

                //ch： 释放采集卡资源  | en: Release the resources of interface
                if (ifInstance != null)
                {
                    ifInstance.Dispose();
                }
            }
        }

        //ch: 打印采集卡信息 | en: print interface informations
        private void PrintInterfaceInfo(List<IInterfaceInfo> ifInfoList)
        {
            int IFIndex = 0;
            foreach (var ifInfo in ifInfoList)
            {
                Console.WriteLine("[Interface {0}]: ", IFIndex);
                Console.WriteLine("TLayerType: " + ifInfo.TLayerType.ToString());
                Console.WriteLine("DisplayName: " + ifInfo.DisplayName);
                Console.WriteLine("InterfaceID: " + ifInfo.InterfaceID);
                Console.WriteLine("SerialNumber: " + ifInfo.SerialNumber);
                Console.WriteLine();
                IFIndex++;
            }
        }

        //ch: 打印设备信息 | en: print device informations
        private void PrintDeviceInfo(List<IDeviceInfo> devInfoList)
        {
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
        }

        static void Main(string[] args)
        {
            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

            InterfaceAndDevice program = new InterfaceAndDevice();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
