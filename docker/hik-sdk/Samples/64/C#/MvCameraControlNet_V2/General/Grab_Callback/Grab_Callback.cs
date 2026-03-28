/*
 * 这个示例演示如何通过回调方式取图（FrameGrabedEvent），回调方式下，SDK内部开启取流线程，通过event方式回调给上层。
 * This program shows how to grab images using event-driven method. The SDK starts a thread to grab images and uses the FrameGrabedEvent to return the images.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;

namespace Grab_Callback
{
    class Grab_Callback
    {
        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
            | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        static void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , ImageSize[{2}], FrameNum[{3}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.Image.ImageSize, e.FrameOut.FrameNum);

            // ch: 手动释放缓存 | en: Dispose frame manually
            e.FrameOut.Dispose();
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
                   Console.WriteLine("No device!");
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

               // ch:设置触发模式为off || en:set trigger mode as off
               ret = device.Parameters.SetEnumValue("TriggerMode", 0);
               if (ret != MvError.MV_OK)
               {
                   Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                   return;
               }

               //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
               device.StreamGrabber.SetImageNodeNum(5);

               // ch:注册回调函数 | en:Register image callback
               device.StreamGrabber.FrameGrabedEventEx += FrameGrabedEventHandler;
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
            catch(Exception e)
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
