/*
 * 这个示例演示了如何在相机异常断开时重新连接相机。
 * This sample shows how to reconnect the camera when it disconnects abnormally.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using MvCameraControl;

namespace Reconnect
{
    class Reconnect
    {
        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
        | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        static bool _bExit = false;
        static bool _bConnect = false;
        static IDevice _device = null;
        static string _serialNumber;

        static void FrameGrabThread(object obj)
        {
            IStreamGrabber streamGrabber = (IStreamGrabber)obj;

            while (true)
            {
                if (!_bConnect)
                {
                    break;
                }
                IFrameOut frame;

                //ch：获取一帧图像 | en: Get one frame
                int ret = streamGrabber.GetImageBuffer(1000, out frame);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    continue;
                }

                Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);
           
                //ch: 释放图像缓存  | en: Release the image buffer
                streamGrabber.FreeImageBuffer(frame);
            }
        }

        static void ExceptionEventHandler(object sender, DeviceExceptionArgs e)
        {
            if (e.MsgType == DeviceExceptionType.DisConnect)
            {
                Console.WriteLine("Device disconnect!");
                _bConnect = false;
            }
        }

        public static void ReconnectProcess()
        {
            int ret = MvError.MV_OK;

            while (true)
            {
                if (_bConnect)
                {
                    Thread.Sleep(1);
                    continue;
                }

                if (_bExit)
                {
                    break;
                }

                if (_device != null)
                {
                    _device.StreamGrabber.StopGrabbing();
                    _device.Close();
                    _device.Dispose();
                    _device = null;
                }

                Console.WriteLine("connecting, please wait...");
                List<IDeviceInfo> devInfoList;
                // ch:枚举设备 | en:Enum device
                ret = DeviceEnumerator.EnumDevices(devLayerType, out devInfoList);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Enum device failed:{0:x8}", ret);
                    continue;
                }

                if (0 == devInfoList.Count)
                {
                    continue;
                }

                //ch:根据序列号选择相机 | en: Select camera by serial number 
                int devIndex = 0;
                bool findDevice = false;
                foreach (var devInfo in devInfoList)
                {
                    if (_serialNumber == devInfo.SerialNumber)
                    {
                        findDevice = true;
                        break;
                    }
                    devIndex++;
                }

                if (!findDevice)
                {
                    continue;
                }
                // ch:创建设备 | en:Create device
                _device = DeviceFactory.CreateDevice(devInfoList[devIndex]);

                // ch:打开设备 | en:Open device
                ret = _device.Open();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Open device failed:{0:x8}", ret);
                    return;
                }

                _bConnect = true;

                // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
                if (_device is IGigEDevice)
                {
                    int packetSize;
                    ret = (_device as IGigEDevice).GetOptimalPacketSize(out packetSize);
                    if (packetSize > 0)
                    {
                        ret = _device.Parameters.SetIntValue("GevSCPSPacketSize", packetSize);
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

                _device.DeviceExceptionEvent += ExceptionEventHandler;

                Console.WriteLine("connect succeed!");

                // ch:开启抓图 | en: start grab image
                ret = _device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    continue;
                }

                // ch:开启抓图线程 | en: Start the grabbing thread
                Thread GrabThread = new Thread(FrameGrabThread);
                GrabThread.Start(_device.StreamGrabber);
            }
        }

        static void Main(string[] args)
        {
            int ret = MvError.MV_OK;
           

            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

            try
            {
                List<IDeviceInfo> devInfoList;

                // ch:枚举设备 | en:Enum device
                ret = DeviceEnumerator.EnumDevices(devLayerType, out devInfoList);
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
                Console.WriteLine();
                if (devIndex > devInfoList.Count - 1 || devIndex < 0)
                {
                    Console.Write("Input Error!\n");
                    return;
                }

                _serialNumber = devInfoList[devIndex].SerialNumber;

                Thread reconnectThread = new Thread(ReconnectProcess);
                reconnectThread.Start();
                               
                Console.WriteLine("Press enter to exit");
                Console.ReadKey();

                _bConnect = false;
                _bExit = true;
                reconnectThread.Join();

                // ch:关闭设备 | en:Close device
                ret = _device.Close();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Close device failed:{0:x8}", ret);
                    return;
                }

                // ch:销毁设备 | en:Destroy device
                _device.Dispose();
                _device = null;

            }
            catch (Exception e)
            {

                Console.Write("Exception: " + e.Message);
            }
            finally
            {
                // ch:销毁设备 | en:Destroy device
                if (_device != null || ret != MvError.MV_OK)
                {
                    _device.Dispose();
                    _device = null;
                }

                // ch: 反初始化SDK | en: Finalize SDK
                SDKSystem.Finalize();

                Console.WriteLine("Press enter to exit");
                Console.ReadKey();
            }
        }
    }
}
