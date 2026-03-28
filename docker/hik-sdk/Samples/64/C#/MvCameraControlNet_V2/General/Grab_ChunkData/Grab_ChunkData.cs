/*
 * 这个示例演示如何解析相机发送的Chunk信息。
 * This program shows how to parse Chunk information sent by a camera.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MvCameraControl;

namespace Grab_ChunkData
{
    class Grab_ChunkData
    {
        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
          | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        static void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);
            
            // ch: chunkData有两种获取方法，一种是通过遍历获取，另一种是通过chunkId获取 
            // en: There are two ways to get chunkData, one is  traversal, and the other is by chunkId

            //ch:方法1：遍历 | en: Way1:traversal
            foreach (IChunkData chunkData in e.FrameOut.ChunkInfo)
            {
                Console.WriteLine("ChunkInfo:" + "ChunkID[0x{0:x8}],ChunkLen[{1}]", chunkData.ChunkID, chunkData.Length);
            }

            //ch:方法2：通过chunkId | en: Way2: byChunkId
            {
                //ch: 获取宽 | en: Get width
                IChunkData widthChunkData = e.FrameOut.ChunkInfo[0xa5a5010a];
                int width = BitConverter.ToInt32(widthChunkData.Data, 0);

                //ch: 获取高 | en: Get heigth
                IChunkData heightChunkData = e.FrameOut.ChunkInfo[0xa5a5010b];
                int height = BitConverter.ToInt32(heightChunkData.Data, 0);

                Console.WriteLine("ChunkInfo:" + "width[{0}],height[{1}]", width, height);
            }

        }

        static void Main(string[] args)
        {
            int ret = MvError.MV_OK;
            IDevice device = null;

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

                // ch:注册回调函数 | en:Register image callback
                device.StreamGrabber.FrameGrabedEvent += FrameGrabedEventHandler;

                // ch:开启Chunk Mode | en:Open Chunk Mode
                ret = device.Parameters.SetBoolValue("ChunkModeActive", true);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Chunk Mode failed:{0:x8}", ret);
                    return;
                }

                // ch:Chunk Selector设为Exposure | en: Chunk Selector set as Exposure
                ret = device.Parameters.SetEnumValueByString("ChunkSelector", "Exposure");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Chunk Enable failed:{0:x8}", ret);
                    return;
                }

                // ch:开启Chunk Enable | en:Open Chunk Enable
                ret = device.Parameters.SetBoolValue("ChunkEnable", true);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Exposure Chunk failed:{0:x8}", ret);
                    return;
                }

                // ch:Chunk Selector设为Timestamp | en: Chunk Selector set as Timestamp
                ret = device.Parameters.SetEnumValueByString("ChunkSelector", "Timestamp");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Timestamp Chunk failed:{0:x8}", ret);
                    return;
                }

                // ch:开启Chunk Enable | en:Open Chunk Enable
                ret = device.Parameters.SetBoolValue("ChunkEnable", true);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Chunk Enable failed:{0:x8}", ret);
                    return;
                }

                // ch:设置触发模式为off || en:set trigger mode as off
                ret = device.Parameters.SetEnumValue("TriggerMode", 0);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

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
                if (device != null || ret != MvError.MV_OK)
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
