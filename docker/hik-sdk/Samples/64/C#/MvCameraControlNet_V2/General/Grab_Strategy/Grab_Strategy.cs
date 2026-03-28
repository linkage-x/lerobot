/*
 * 这个示例演示了如何使用取图策略
 * This sample demonstrates how to use the image acquisition strategy
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;
using System.Threading;

namespace Grab_Strategy
{
    class Grab_Strategy
    {
        //Only the GigE and USB device support the grab strategy now!
        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice;

        static volatile bool _grabThreadExit = false;
        static void FrameGrabThread(object obj)
        {
            IStreamGrabber streamGrabber = (IStreamGrabber)obj;

            while (!_grabThreadExit)
            {
                IFrameOut frame;

                //ch：获取一帧图像 | en: Get one frame
                int ret = streamGrabber.GetImageBuffer(1000, out frame);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    continue;
                }

                Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                //Do something


                //ch: 释放图像缓存  | en: Release the image buffer
                streamGrabber.FreeImageBuffer(frame);
            }
        }

        static void StrategyOneByOne(IDevice device)
        {
            int ret = MvError.MV_OK;

            // ch:开启抓图 | en: start grab image
            ret = device.StreamGrabber.StartGrabbing(StreamGrabStrategy.OneByOne);
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                return;
            }

            // ch:软触发5次 | en: 5 software triggers
            for (UInt32 i = 0; i < 5; i++)
            {
                //ch: 软触发 | en: Software trigger
                ret = device.Parameters.SetCommandValue("TriggerSoftware");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Send Trigger Software command fail:{0:x8}", ret);
                }
                else
                {
                    Console.WriteLine("Send Trigger Software command:{0}", i);
                }

                Thread.Sleep(500);//如果帧率过小或TriggerDelay很大，可能会出现软触发命令没有全部起效而导致取不到数据的情况
            }


            //ch: 获取到连续的5帧图像 | en: Obtain a continuous sequence of 5 frames
            while (true)
            {
                IFrameOut frame;
                ret = device.StreamGrabber.GetImageBuffer(10, out frame);
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                    //Do something

                    //ch: 释放图像缓存  | en: Release the image buffer
                    device.StreamGrabber.FreeImageBuffer(frame);
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    break;
                }
            }

            // ch:停止抓图 | en:Stop grabbing
            ret = device.StreamGrabber.StopGrabbing();
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                return;
            }
        }


        static void StrategyLastImageOnly(IDevice device)
        {
            int ret = MvError.MV_OK;

            // ch:开启抓图 | en: start grab image
            ret = device.StreamGrabber.StartGrabbing(StreamGrabStrategy.LatestImageOnly);
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                return;
            }

            // ch:软触发5次 | en: 5 software triggers
            for (UInt32 i = 0; i < 5; i++)
            {
                //ch: 软触发 | en: Software trigger
                ret = device.Parameters.SetCommandValue("TriggerSoftware");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Send Trigger Software command fail:{0:x8}", ret);
                }
                else
                {
                    Console.WriteLine("Send Trigger Software command:{0}", i);
                }

                Thread.Sleep(500);//如果帧率过小或TriggerDelay很大，可能会出现软触发命令没有全部起效而导致取不到数据的情况
            }


            //ch：获取到最新一帧图像  | en：Obtain the latest frame
            while (true)
            {
                IFrameOut frame;
                ret = device.StreamGrabber.GetImageBuffer(10, out frame);
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                    //Do something

                    //ch: 释放图像缓存  | en: Release the image buffer
                    device.StreamGrabber.FreeImageBuffer(frame);
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    break;
                }
            }

            // ch:停止抓图 | en:Stop grabbing
            ret = device.StreamGrabber.StopGrabbing();
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                return;
            }
        }

        static void StrategyLastImages(IDevice device)
        {
            int ret = MvError.MV_OK;

            //ch: 设置输出2帧图像 | Set out put queue to 2
            ret = device.StreamGrabber.SetOutputQueueSize(2);

            // ch:开启抓图 | en: start grab image
            ret = device.StreamGrabber.StartGrabbing(StreamGrabStrategy.LatestImages);
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                return;
            }

            // ch:软触发5次 | en: 5 software triggers
            for (UInt32 i = 0; i < 5; i++)
            {
                //ch: 软触发 | en: Software trigger
                ret = device.Parameters.SetCommandValue("TriggerSoftware");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Send Trigger Software command fail:{0:x8}", ret);
                }
                else
                {
                    Console.WriteLine("Send Trigger Software command:{0}", i);
                }

                Thread.Sleep(500);//如果帧率过小或TriggerDelay很大，可能会出现软触发命令没有全部起效而导致取不到数据的情况
            }

            //ch：获取到最新2帧图像  | en：Obtain the latest two frames
            while (true)
            {
                IFrameOut frame;
                ret = device.StreamGrabber.GetImageBuffer(10, out frame);
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                    //Do something

                    //ch: 释放图像缓存  | en: Release the image buffer
                    device.StreamGrabber.FreeImageBuffer(frame);
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    break;
                }
            }

            // ch:停止抓图 | en:Stop grabbing
            ret = device.StreamGrabber.StopGrabbing();
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                return;
            }
        }

        static void StrategyUpcomingImage(IDevice device)
        {
            int ret = MvError.MV_OK;

            // ch:开启抓图 | en: start grab image
            ret = device.StreamGrabber.StartGrabbing(StreamGrabStrategy.UpcomingImage);
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                return;
            }

            // ch:软触发5次 | en: 5 software triggers
            for (UInt32 i = 0; i < 5; i++)
            {
                //ch: 软触发 | en: Software trigger
                ret = device.Parameters.SetCommandValue("TriggerSoftware");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Send Trigger Software command fail:{0:x8}", ret);
                }
                else
                {
                    Console.WriteLine("Send Trigger Software command:{0}", i);
                }

                Thread.Sleep(500);//如果帧率过小或TriggerDelay很大，可能会出现软触发命令没有全部起效而导致取不到数据的情况
            }

            Thread triggerThread = new Thread(() =>
                {
                    // ch: 3秒后触发一次 | en: trigger after 3s
                    Thread.Sleep(3000);
                    device.Parameters.SetCommandValue("TriggerSoftware");
                    Console.WriteLine("TriggerThread：Send Trigger Software command");
                });
            triggerThread.Start();

            //ch：获取到3秒后触发的那一帧  | en: Retrieve the frame triggered 3 seconds later
            while (true)
            {
                IFrameOut frame;
                ret = device.StreamGrabber.GetImageBuffer(5000, out frame);
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                    //Do something

                    //ch: 释放图像缓存  | en: Release the image buffer
                    device.StreamGrabber.FreeImageBuffer(frame);

                    break;
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                }
            }

            // ch:停止抓图 | en:Stop grabbing
            ret = device.StreamGrabber.StopGrabbing();
            if (ret != MvError.MV_OK)
            {
                Console.WriteLine("Stop grabbing failed:{0:x8}", ret);
                return;
            }
        }

        static void Main(string[] args)
        {
            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

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

                // ch:设置为软触发模式 | en:Set Trigger Mode and Set Trigger Source
                ret = device.Parameters.SetEnumValueByString("TriggerMode", "On");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

                ret = device.Parameters.SetEnumValueByString("TriggerSource", "Software");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Trigger Source failed:{0:x8}", ret);
                    return;
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                device.StreamGrabber.SetImageNodeNum(10);


                Console.WriteLine("\n**************************************************************************");
                Console.WriteLine("* 0.GrabStrategy_OneByOne;       1.GrabStrategy_LatestImagesOnly;  *");
                Console.WriteLine("* 2.GrabStrategy_LatestImages;   3.GrabStrategy_UpcomingImage;     *");
                Console.WriteLine("**************************************************************************");

                Console.Write("Please Input Grab Strategy:");
                Int32 nGrabStrategy = 0;
                try
                {
                    nGrabStrategy = (Int32)Convert.ToInt32(Console.ReadLine());
                }
                catch
                {
                    Console.Write("Invalid Input!");
                    return;
                }

                // ch:U3V相机不支持MV_GrabStrategy_UpcomingImage | en:U3V device not support UpcomingImage
                if (nGrabStrategy == (Int32)StreamGrabStrategy.UpcomingImage
                    && device is IUSBDevice)
                {
                    Console.Write("U3V device not support UpcomingImage");
                    return;
                }

                if (nGrabStrategy == (Int32)StreamGrabStrategy.OneByOne)
                {
                    StrategyOneByOne(device);
                }
                else if (nGrabStrategy == (Int32)StreamGrabStrategy.LatestImageOnly)
                {
                    StrategyLastImageOnly(device);
                }
                else if (nGrabStrategy == (Int32)StreamGrabStrategy.LatestImages)
                {
                    StrategyLastImages(device);
                }
                else if (nGrabStrategy == (Int32)StreamGrabStrategy.UpcomingImage)
                {
                    StrategyUpcomingImage(device);
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

                // ch: 反初始化SDK | en: Finalize SDK
                SDKSystem.Finalize();

                Console.WriteLine("Press enter to exit");
                Console.ReadKey();
            }
        }
    }
}
