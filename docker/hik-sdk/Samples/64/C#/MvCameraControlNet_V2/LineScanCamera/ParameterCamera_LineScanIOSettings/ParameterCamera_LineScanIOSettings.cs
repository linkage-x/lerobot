/*
 * 这个示例演示配置线阵相机的IO参数，回调方式下，SDK内部开启取流线程，通过event方式回调给上层。部分节点的设置需要搭配采集卡和设备的ScanMode。
 * This program shows demonstrate configuring the IO parameters for LineScan camera. The SDK starts a thread to grab images and uses the FrameGrabedEvent to return the images.
 */

using MvCameraControl;
using System;
using System.Collections.Generic;

namespace ParameterCamera_LineScanIOSettings
{
    class ParameterCamera_LineScanIOSettings
    {
        IDevice device = null;

        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
            | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);
        }

        //ch: 检查属性节点的访问模式 | en: Check the access mode of the feature node
        bool CheckFeatureNodeAccess(string nodeName)
        {
            XmlAccessMode nMode;
            int ret = device.Parameters.GetNodeAccessMode(nodeName, out nMode);
            if (ret != MvError.MV_OK)
            {
                return false;
            }
            if(nMode == XmlAccessMode.RO || nMode == XmlAccessMode.RW || nMode == XmlAccessMode.WO)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public void Run()
        {
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

                // ch:选择设备 | en:Select a device
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
                Console.WriteLine("Open device success");

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

                Console.WriteLine("");
                Console.WriteLine("Please Input trigger selector index: 0-FrameTrigger, 1-LineTrigger\n");
                uint triggerSelector = Convert.ToUInt32(Console.ReadLine());

                switch (triggerSelector)
                {
                    case 0:
                        {
                            // ch:设置ScanMode为FrameScan| en:Set ScanMode to FrameScan
                            ret = device.Parameters.SetEnumValueByString("ScanMode", "FrameScan");
                            if (ret == MvError.MV_OK)
                            {
                                Console.WriteLine("Set Frame Scan Mode", ret.ToString("X"));
                            }

                            // ch: 判断FrameTriggerControl是否可读 | en: Check if FrameTriggerControl is readable
                            if (CheckFeatureNodeAccess("FrameTriggerControl"))
                            {
                                // ch:设置触发模式为on | en:Set trigger mode as On
                                ret = device.Parameters.SetBoolValue("FrameTriggerMode", true);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Frame Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Frame Trigger Mode to On");

                                // ch:设置触发源为Line0 | en:Set trigger source as Line0
                                ret = device.Parameters.SetEnumValue("FrameTriggerSource", 0);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Frame Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Frame Trigger Source to Line0");
                            }
                            else
                            {
                                // ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
                                ret = device.Parameters.SetEnumValue("TriggerSelector", 6);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Selector fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Selector to FrameBurstStart");

                                // ch:设置触发模式为on | en:Set trigger mode as On
                                ret = device.Parameters.SetEnumValue("TriggerMode", 1);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Mode to On");

                                // ch:设置触发源为Line0 | en:Set trigger source as Line0
                                ret = device.Parameters.SetEnumValue("TriggerSource", 0);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Source to Line0");
                            }
                        }
                        break;
                    case 1:
                        {
                            // ch: 判断LineTriggerControl是否可读 | en: Check if LineTriggerControl is readable
                            if (CheckFeatureNodeAccess("LineTriggerControl"))
                            {
                                // ch:设置触发模式为on | en:Set trigger mode as On
                                ret = device.Parameters.SetBoolValue("LineTriggerMode", true);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Line Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Line Trigger Mode to On");

                                // ch:设置触发源为EncoderModuleOut | en:Set trigger source as EncoderModuleOut
                                ret = device.Parameters.SetEnumValue("LineTriggerSource", 6);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Line Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Line Trigger Source to EncoderModuleOut");

                                // ch:设置编码器选项为Encoder0 | en:Set encoder selector as Encoder0
                                ret = device.Parameters.SetEnumValue("EncoderSelector", 0);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Encoder Selector fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Encoder Selector to Encoder0");

                                // ch:设置编码器数据源A为Line1 | en:Set encoder source A as Line1
                                ret = device.Parameters.SetEnumValue("EncoderSourceA", 1);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set encoder sourceA fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set EncoderSourceA to Line1");

                                // ch:设置编码器数据源B为Line3 | en:Set encoder source B as Line3
                                ret = device.Parameters.SetEnumValue("EncoderSourceB", 3);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set encoder sourceB fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set EncoderSourceB to Line3");
                            }
                            else
                            {
                                // ch:设置触发选项为LineStart | en:Set trigger selector as LineStart
                                ret = device.Parameters.SetEnumValue("TriggerSelector", 9);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Selector fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Selector to LineStart");

                                // ch:设置触发模式为on | en:Set trigger mode as On
                                ret = device.Parameters.SetEnumValue("TriggerMode", 1);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Mode to On");

                                // ch:设置触发源为EncoderModuleOut | en:Set trigger source as EncoderModuleOut
                                ret = device.Parameters.SetEnumValue("TriggerSource", 6);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Trigger Source to EncoderModuleOut");

                                // ch:设置编码器选项为Encoder0 | en:Set encoder selector as Encoder0
                                ret = device.Parameters.SetEnumValue("EncoderSelector", 0);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set Encoder Selector fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set Encoder Selector to Encoder0");

                                // ch:设置编码器数据源A为Line1 | en:Set encoder source A as Line1
                                ret = device.Parameters.SetEnumValue("EncoderSourceA", 1);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set encoder sourceA fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set EncoderSourceA to Line1");

                                // ch:设置编码器数据源B为Line3 | en:Set encoder source B as Line3
                                ret = device.Parameters.SetEnumValue("EncoderSourceB", 3);
                                if (ret != MvError.MV_OK)
                                {
                                    Console.WriteLine("Set encoder sourceB fail! nRet {0}\n", ret.ToString("X"));
                                    break;
                                }
                                Console.WriteLine("Set EncoderSourceB to Line3");
                            }
                        }
                        break;
                    default:
                        Console.WriteLine("Input error!");
                        break;
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

                Console.WriteLine("Press enter to Stop Grabbing");
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

                // ch:销毁设备 | en:Destroy device
                device.Dispose();
                device = null;
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

            ParameterCamera_LineScanIOSettings program = new ParameterCamera_LineScanIOSettings();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
