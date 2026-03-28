/*
 * 这个示例演示了从相机中导入导出配置文件。
 * This sample shows how to importing and exporting configuration files from a camera.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;

namespace ParameterCamera_LoadAndSave
{
    class ParameterCamera_LoadAndSave
    {
        private const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
           | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        public void Run()
        {
            IDevice device = null;

            try
            {
                List<IDeviceInfo> devInfoList;

                // ch:枚举设备 | en:Enum device
                int ret = DeviceEnumerator.EnumDevices(devLayerType, out devInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Enum device failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Enum device count : {0}", devInfoList.Count);

                if (0 == devInfoList.Count)
                {
                    return;
                }

                // ch:打印设备信息 en:Print device info
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

                Console.Write("Please input index(0-{0:d}):", devInfoList.Count - 1);

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

                // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
                if (device is IGigEDevice)
                {
                    int packetSize;
                    ret = (device as IGigEDevice).GetOptimalPacketSize(out packetSize);
                    if (packetSize > 0)
                    {
                        ret = device.Parameters.SetIntValue("GevSCPSPacketSize", packetSize);
                        if (ret != MvError.MV_OK)
                        {
                            Console.WriteLine("Warning: Set Packet Size failed {0:x8}", ret);
                        }
                        else
                        {
                            Console.WriteLine("Set PacketSize to {0}", packetSize);
                        }
                    }
                    else
                    {
                        Console.WriteLine("Warning: Get Packet Size failed {0:x8}", ret);
                    }
                }

                Console.WriteLine("Start export the camera properties to the file");
                Console.WriteLine("Wait......");
                // ch:将相机属性导出到文件中
                // en:Export the camera properties to the file
                ret = device.Parameters.FeatureSave("CameraFile.mfs");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("FeatureSave failed!");
                    return;
                }

                Console.WriteLine("Finish export the camera properties to the file\n");

                Console.WriteLine("Start import the camera properties from the file");
                Console.WriteLine("Wait......");
                // ch:从文件中导入相机属性
                // en:Import the camera properties from the file
                ret = device.Parameters.FeatureLoad("CameraFile.mfs");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("FeatureLoad failed!");
                    return;
                }

                Console.WriteLine("Finish import the camera properties from the file");

               
                // ch:关闭设备 | en:Close device
                ret = device.Close();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Close device failed:{0:x8}", ret);
                    return;
                }

                // ch:销毁设备 | en:Destroy device
                device.Dispose();
            }
            catch (Exception e)
            {
                Console.Write("Exception: " + e.Message);
            }
            finally
            {
                // ch:销毁设备 | en:Destroy device
                if (device != null)
                {
                    device.Dispose();
                    device = null;
                }

            }
        }


        static void Main(string[] args)
        {
            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

            ParameterCamera_LoadAndSave program = new ParameterCamera_LoadAndSave();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
