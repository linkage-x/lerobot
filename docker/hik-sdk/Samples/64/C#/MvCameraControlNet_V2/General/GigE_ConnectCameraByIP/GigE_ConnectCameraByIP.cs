/*
 * 这个示例演示如何通过IP连接GigE相机。
 * This program shows how to connect a GigE camera through IP.
 */

using MvCameraControl;
using System;

namespace GigE_ConnectCameraByIP
{
    class GigE_ConnectCameraByIP
    {
        static void Main(string[] args)
        {

            int result = MvError.MV_OK;
            IDevice device = null;
            int packetSize;
          
            try
            {
                //ch: 初始化SDK |  en: Initialize SDK
                SDKSystem.Initialize();

                // ch:需要连接的相机ip(根据实际填充) 
               //en:The camera IP that needs to be connected (based on actual padding)
                Console.Write("Please input Device Ip : ");
                string deviceIp = Convert.ToString(Console.ReadLine());

                // ch:相机对应的网卡ip(根据实际填充)
                // en:The pc IP that needs to be connected (based on actual padding)
                Console.Write("Please input Net Export Ip : ");
                string netExport = Convert.ToString(Console.ReadLine());

                //ch: 创建设备 | en: Create device
                device = DeviceFactory.CreateDeviceByIp(deviceIp, netExport);
                if(null == device)
                {
                    Console.WriteLine("Create device failed!");
                    return;
                }

                //ch: 打开设备 | en:Open device
                result = device.Open();
                if (MvError.MV_OK != result)
                {
                    Console.WriteLine("Open device failed:{0:x8}", result);
                    return;
                }

                //ch: 判断是否为gige设备 | en: Determine whether it is a GigE device
                if (device is IGigEDevice)
                {
                    //ch: 转换为gigE设备 | en: Convert to Gige device
                    IGigEDevice gigEDevice = (IGigEDevice)device;


                    // ch:探测网络最佳包大小(只对GigE相机有效) 
                    // en:Detection network optimal package size(It only works for the GigE camera)
                    result = gigEDevice.GetOptimalPacketSize(out packetSize);
                    if (MvError.MV_OK != result)
                    {
                        Console.WriteLine("Warning: Get Packet Size failed {0:x8}", result);
                    }
                    else
                    {
                       
                        result = gigEDevice.Parameters.SetIntValue("GevSCPSPacketSize", packetSize);
                        if (MvError.MV_OK != result)
                        {
                            Console.WriteLine("Warning: Set Packet Size failed {0:x8}", result);
                        }
                    }
                }
                else
                {
                    Console.WriteLine(" Device is not gigEDevice!");
                    return;
                }

                // ch:设置触发模式为off || en:set trigger mode as off
                result = device.Parameters.SetEnumValue("TriggerMode", 0);
                if (MvError.MV_OK != result)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", result);
                    return;
                }

                // ch:开启抓图 || en: start grab image
                result = device.StreamGrabber.StartGrabbing();
                if (MvError.MV_OK != result)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", result);
                    return;
                }

                int count = 0;
                IFrameOut frameOut;
                while(count++ != 10)
                {
                    // ch:获取一帧图像 | en:Get one image
                    result = device.StreamGrabber.GetImageBuffer(1000, out frameOut);
                    if (MvError.MV_OK == result)
                    {
                        Console.WriteLine("Get Image Buffer: Width[{0}] , Height[{1}] , FrameNum[{2}]"
                            , frameOut.Image.Width, frameOut.Image.Height, frameOut.FrameNum);

                        ImageFormatInfo info = new ImageFormatInfo();
                        info.FormatType = ImageFormatType.Jpeg;
                        info.JpegQuality = 80;

                        string filePath = "Image" + "_w_" + frameOut.Image.Width + "_h" + frameOut.Image.Height + "_p" + frameOut.Image.PixelType.ToString() + "_" + frameOut.FrameNum;
                        filePath += "." + info.FormatType;

                        //ch: 保存图像到文件 | en: Save image to file
                        result = device.ImageSaver.SaveImageToFile(filePath, frameOut.Image, info, CFAMethod.Equilibrated);
                        if (MvError.MV_OK != result)
                        {
                            Console.WriteLine("Save Image failed:{0:x8}", result);                       
                        }

                        //ch: 释放图像缓存 | en: Release image buffer
                        device.StreamGrabber.FreeImageBuffer(frameOut);
                    }
                    else
                    {
                        Console.WriteLine("Get Image failed:{0:x8}", result);
                    }
                }

                // ch:停止抓图 | en:Stop grabbing
                result = device.StreamGrabber.StopGrabbing();
                if (MvError.MV_OK != result)
                {
                    Console.WriteLine("Stop grabbing failed:{0:x8}", result);
                    return;
                }

                // ch:关闭设备 | en:Close device
                result = device.Close();
                if (MvError.MV_OK != result)
                {
                    Console.WriteLine("Close device failed:{0:x8}", result);
                    return;
                }

                // ch:销毁设备 | en:Destroy device
                device.Dispose();

            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                if (MvError.MV_OK != result)
                {
                    // ch:销毁设备 | en:Destroy device
                    device.Dispose();
                }

                // ch: 反初始化SDK | en: Finalize SDK
                SDKSystem.Finalize();

                Console.WriteLine("Press enter to exit");
                Console.ReadKey();
            }
      
        }
    }
}
