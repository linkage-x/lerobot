/*
 * 这个示例演示了如何使用对比度调节功能
 * This sample demonstrates how to use the image contrast function of ImageProcess
 */

using MvCameraControl;
using System;
using System.Collections.Generic;

namespace Image_Contrast
{
    class Image_Contrast
    {
        static void Main(string[] args)
        {
            DeviceTLayerType enumTLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice
     | DeviceTLayerType.MvGenTLGigEDevice | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLCameraLinkDevice | DeviceTLayerType.MvGenTLXoFDevice;


            int result = MvError.MV_OK;
            IDevice device = null;
            List<IDeviceInfo> deviceInfos;
            int packetSize;

            try
            {
                //ch: 初始化SDK |  en: Initialize SDK
                SDKSystem.Initialize();

                result = DeviceEnumerator.EnumDevices(enumTLayerType, out deviceInfos);
                if (result != MvError.MV_OK)
                {
                    Console.WriteLine("Enumerate device failed, result: {0:x8}", result);
                    return;
                }

                if (deviceInfos.Count == 0)
                {
                    Console.WriteLine("No device");
                    return;
                }

                // ch:打印设备信息 en:Print device info
                int devIndex = 0;
                foreach (var devInfo in deviceInfos)
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

                // ch:需要连接的相机索引 || en:Select a device that want to connect
                Console.Write("Please input index(0-{0:d}):", deviceInfos.Count - 1);

                devIndex = Convert.ToInt32(Console.ReadLine());

                if (devIndex > deviceInfos.Count - 1 || devIndex < 0)
                {
                    Console.Write("Input Error!\n");
                    return;
                }

                //ch: 创建设备 | en: Create device
                device = DeviceFactory.CreateDevice(deviceInfos[devIndex]);
                if (device == null)
                {
                    Console.WriteLine("Create device failed!");
                    return;
                }

                //ch: 打开设备 | en:Open device
                result = device.Open();
                if (result != MvError.MV_OK)
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

                IFrameOut frameOut;
                // ch:获取一帧图像 | en:Get one image
                result = device.StreamGrabber.GetImageBuffer(1000, out frameOut);
                if (MvError.MV_OK == result)
                {
                    Console.WriteLine("Get Image Buffer: Width[{0}] , Height[{1}] , FrameNum[{2}]"
                        , frameOut.Image.Width, frameOut.Image.Height, frameOut.FrameNum);

                    IImage inputImage = frameOut.Image;
                    IImage outImage;

                    // ch:对比度值，[1, 10000] | en:Image Contrast Factor[1, 10000]
                    uint contrastFactor = 300;

                    // ch:对比度调节 | en:Image Contrast Process
                    result = device.ImageProcess.ImageContrast(inputImage, out outImage, contrastFactor);
                    if (result != MvError.MV_OK)
                    {
                        Console.WriteLine("Image Contrast failed:{0:x8}", result);
                        return;
                    }
                    Console.WriteLine("Image Contrast success!");

                    ImageFormatInfo info = new ImageFormatInfo();
                    info.FormatType = ImageFormatType.Bmp;

                    string inputFilePath = string.Format("InputImage.{0}", info.FormatType);
                    string outputFilePath = string.Format("OutputImage_ContrastFactor{0}.{1}", contrastFactor, info.FormatType);

                    //ch: 保存图像到文件 | en: Save image to file
                    device.ImageSaver.SaveImageToFile(inputFilePath, frameOut.Image, info, CFAMethod.Equilibrated);
                    Console.WriteLine("Save inputImage: {0}!", inputFilePath);

                    device.ImageSaver.SaveImageToFile(outputFilePath, outImage, info, CFAMethod.Equilibrated);
                    Console.WriteLine("Save OutputImage: {0}!", outputFilePath);

                    //ch: 图像使用完及时释放，防止内存快速上涨导致频繁GC |en：Release image promptly to prevent rapid memory increase leading to frequent GC.
                    outImage.Dispose();

                    //ch: 释放图像缓存 | en: Release image buffer
                    device.StreamGrabber.FreeImageBuffer(frameOut);
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", result);
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
                device = null;

            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            finally
            {
                if (device != null)
                {
                    // ch:销毁设备 | en:Destroy device
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
