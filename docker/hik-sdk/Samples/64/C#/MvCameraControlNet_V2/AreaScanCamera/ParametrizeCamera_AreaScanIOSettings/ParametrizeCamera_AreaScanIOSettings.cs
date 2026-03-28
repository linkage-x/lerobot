/*
 * 这个示例演示了从相机中获取配置文件。
 * This sample shows how to obtain configuration files from a camera.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;
using System.Threading;

namespace ParameterCamera_AreaScanIOSettings
{
    class ParameterCamera_AreaScanIOSettings
    {
        private const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
           | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;


        void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , ImageSize[{2}], FrameNum[{3}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.Image.ImageSize, e.FrameOut.FrameNum);
        }

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

                // ch:打开设备 | en:Open device
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


                //Ch: 设置Line0 IO输入(也可选择Line2作为输入, 延迟更低) | en: Set Line0 to IO input(You can also choose Line2 as Input IO, which has Lower latency)
                Console.WriteLine("Set IO input...");
                {
                    // ch:设置触发模式为on | en:Set trigger mode as on
                    ret = device.Parameters.SetEnumValueByString("TriggerMode", "On");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set trigger mode fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置触发源为Line0 | en:Set trigger source as Line0
                    ret = device.Parameters.SetEnumValueByString("TriggerSource", "Line0");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set trigger source fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置触发沿为RisingEdge | en: Set TriggerActivation as RisingEdge
                    ret = device.Parameters.SetEnumValueByString("TriggerActivation", "RisingEdge");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set trigger activation RisingEdge fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置触发延迟 | en: Set TriggerDelay
                    ret = device.Parameters.SetFloatValue("TriggerDelay", 0);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set trigger delay fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:关闭触发缓存、如需要可开启 | en: Turn off TriggerCacheEnable 
                    ret = device.Parameters.SetBoolValue("TriggerCacheEnable", false);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set trigger cache enable fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:切换LineSelector 为Line0 | en: Set LineSelector as Line0
                    ret = device.Parameters.SetEnumValueByString("LineSelector", "Line0");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line selector fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置Line0 滤波时间(us)、误触发可适当加大 | Set Line0 LineDebouncerTime(us)
                    ret = device.Parameters.SetIntValue("LineDebouncerTime", 50);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line debouncer time fail! nRet {0:x8}\n", ret);
                        return;
                    }
                }

                //ch: 设置Line1 IO输出(也可选择Line2作为输出, 延迟更低), 用于控制外部光源、等设备 | en: Set Line1 as IO output (You can also choose Line2 as Output IO, which has Lower latency )
                Console.WriteLine("Set IO output...");
                {
                    // ch:切换LineSelector 为Line1 | en:Set LineSelector as Line1
                    ret = device.Parameters.SetEnumValueByString("LineSelector", "Line1");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line selector fail! nRet {0:x8}", ret);
                        return;
                    }
                    //ch: 输出源选择曝光开始 |en:Set LineSource as ExposureStartActive
                    ret = device.Parameters.SetEnumValueByString("LineSource", "ExposureStartActive");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set LineSource fail! nRet {0:x8}", ret);
                        return;
                    }
                    // ch:开启输出使能| en: Turn on StrobeEnable
                    ret = device.Parameters.SetBoolValue("StrobeEnable", true);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set strobe enable fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置输出线路持续时间（us） | en: Set StrobeLineDuration(us)
                    ret = device.Parameters.SetIntValue("StrobeLineDuration", 0);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line strobe line duration fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:设置输出线路持续时间（us） | en: Set StrobeLineDuration(us)
                    ret = device.Parameters.SetIntValue("StrobeLineDelay", 0);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line strobe line delay fail! nRet {0:x8}", ret);
                        return;
                    }

                    // ch:输出线路预延迟（us） | en: Set StrobeLinePreDelay(us)
                    ret = device.Parameters.SetIntValue("StrobeLinePreDelay", 0);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set line strobe line pre-delay fail! nRet {0:x8}", ret);
                        return;
                    }
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                device.StreamGrabber.SetImageNodeNum(5);

                // ch:注册回调函数 | en:Register image callback
                device.StreamGrabber.FrameGrabedEvent += FrameGrabedEventHandler;
                // ch:开启抓图 || en: start grab image
                ret = device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Press enter to exit");
                Console.ReadLine();


                // ch:停止抓图 | en:Stop grabbing
                ret = device.StreamGrabber.StopGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                    return;
                }

                // ch:关闭设备 | en:Close device
                ret = device.Close();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Close device failed:{0:x8}", ret);
                    return;
                }
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

            ParameterCamera_AreaScanIOSettings program = new ParameterCamera_AreaScanIOSettings();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
