/*
 * 这个示例演示线阵相机如何通过软触发命令获取图像
 * This simple shows how a line-scan camera acquires images through software trigger commands.
 */

using MvCameraControl;
using System;
using System.Collections.Generic;
using System.Threading;

namespace LineScanSoftwareTrigger
{
    class LineScanSoftwareTrigger
    {
        IDevice device = null;
        string triggerCmd="";

        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
            | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        volatile bool _cmdThreadExit = false;
        void SoftwareTriggerCommandThread(object obj)
        {
            IParameters parameters = (IParameters)obj;

            while (!_cmdThreadExit)
            {
                //ch：发送软触发命令 | en: set software trigger command
                int ret = parameters.SetCommandValue(triggerCmd);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set trigger command fail:{0:x8}", ret);
                    continue;
                }

                Thread.Sleep(1000);
            }
        }

        void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);
        }

        //ch：检查节点是否可访问 | en: check if node is accessible
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

                ret = device.Parameters.SetEnumValueByString("ScanMode", "FrameScan");
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Set Frame Scan Mode ");
                }

                // ch: 检查FrameTriggerControl是否可读 | en: Check if FrameTriggerControl is readable
                if (CheckFeatureNodeAccess("FrameTriggerControl"))
                {
                    // ch:设置触发模式为on | en:Set trigger mode as On
                    ret = device.Parameters.SetBoolValue("FrameTriggerMode", true);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set Frame Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                        return;
                    }
                    Console.WriteLine("Set Frame Trigger Mode to On");

                    // ch:设置触发源为Line0 | en:Set trigger source as software
                    ret = device.Parameters.SetEnumValueByString("FrameTriggerSource", "Software");
                    if (ret != MvError.MV_OK)
                    {
                         Console.WriteLine("Set Frame Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                         return;
                    }
                    Console.WriteLine("Set Frame Trigger Source to software");

                    triggerCmd = "FrameTriggerSoftware";
                }
                else
                {
                    // ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
                    ret = device.Parameters.SetEnumValue("TriggerSelector", 6);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set Trigger Selector fail! nRet {0}\n", ret.ToString("X"));
                        return;
                    }
                    Console.WriteLine("Set Trigger Selector to FrameBurstStart");

                    // ch:设置触发模式为on | en:Set trigger mode as On
                    ret = device.Parameters.SetEnumValue("TriggerMode", 1);
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set Trigger Mode fail! nRet {0}\n", ret.ToString("X"));
                        return;
                    }
                    Console.WriteLine("Set Trigger Mode to On");

                    // ch:设置触发源为Line0 | en:Set trigger source as Software
                    ret = device.Parameters.SetEnumValueByString("TriggerSource", "Software");
                    if (ret != MvError.MV_OK)
                    {
                        Console.WriteLine("Set Trigger Source fail! nRet {0}\n", ret.ToString("X"));
                        return;
                    }
                    Console.WriteLine("Set Trigger Source to software");

                    triggerCmd = "TriggerSoftware";
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

                // ch:开启执行触发命令线程 | en: Start set trigger command thread
                Thread CommandThread = new Thread(SoftwareTriggerCommandThread);
                CommandThread.Start(device.Parameters);

                Console.WriteLine("Press enter to Stop Grabbing");
                Console.ReadLine();

                //ch: 通知线程退出 | en: Notify the thread to exit
                _cmdThreadExit = true;
                CommandThread.Join();


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

            LineScanSoftwareTrigger program = new LineScanSoftwareTrigger();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
