/*
 * 这个示例演示了如何使用格式转换功能
 * This sample demonstrates how to use the PixelTypeConverter function
 */

using MvCameraControl;
using System;
using System.Collections.Generic;
using System.IO;

namespace Image_ConvertPixelType
{
    class Image_ConvertPixelType
    {
        static bool IsMonoPixelFormat(MvGvspPixelType enType)
        {
            switch (enType)
            {
                case MvGvspPixelType.PixelType_Gvsp_Mono10:
                case MvGvspPixelType.PixelType_Gvsp_Mono10_Packed:
                case MvGvspPixelType.PixelType_Gvsp_Mono12:
                case MvGvspPixelType.PixelType_Gvsp_Mono12_Packed:
                    return true;
                default:
                    return false;
            }
        }

        static bool IsColorPixelFormat(MvGvspPixelType enType)
        {
            switch (enType)
            {
                case MvGvspPixelType.PixelType_Gvsp_BGR8_Packed:
                case MvGvspPixelType.PixelType_Gvsp_YUV422_Packed:
                case MvGvspPixelType.PixelType_Gvsp_YUV422_YUYV_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerGR8:
                case MvGvspPixelType.PixelType_Gvsp_BayerRG8:
                case MvGvspPixelType.PixelType_Gvsp_BayerGB8:
                case MvGvspPixelType.PixelType_Gvsp_BayerBG8:
                case MvGvspPixelType.PixelType_Gvsp_BayerRBGG8:
                case MvGvspPixelType.PixelType_Gvsp_BayerGB10:
                case MvGvspPixelType.PixelType_Gvsp_BayerGB10_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerBG10:
                case MvGvspPixelType.PixelType_Gvsp_BayerBG10_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerRG10:
                case MvGvspPixelType.PixelType_Gvsp_BayerRG10_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerGR10:
                case MvGvspPixelType.PixelType_Gvsp_BayerGR10_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerGB12:
                case MvGvspPixelType.PixelType_Gvsp_BayerGB12_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerBG12:
                case MvGvspPixelType.PixelType_Gvsp_BayerBG12_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerRG12:
                case MvGvspPixelType.PixelType_Gvsp_BayerRG12_Packed:
                case MvGvspPixelType.PixelType_Gvsp_BayerGR12:
                case MvGvspPixelType.PixelType_Gvsp_BayerGR12_Packed:
                    return true;
                default:
                    return false;
            }
        }

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
                    Console.WriteLine("Input Error!\n");
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
                    MvGvspPixelType dstPixelType = MvGvspPixelType.PixelType_Gvsp_Undefined;

                    if (IsColorPixelFormat(frameOut.Image.PixelType))
                    {
                        dstPixelType = MvGvspPixelType.PixelType_Gvsp_RGB8_Packed;
                    }
                    else if (IsMonoPixelFormat(frameOut.Image.PixelType))
                    {
                        dstPixelType = MvGvspPixelType.PixelType_Gvsp_Mono8;
                    }
                    else
                    {
                        Console.WriteLine("Don't need to convert!");
                    }

                    if (dstPixelType != MvGvspPixelType.PixelType_Gvsp_Undefined)
                    {
                        // ch:像素格式转换 | en:Pixel type convert 
                        result = device.PixelTypeConverter.ConvertPixelType(inputImage, out outImage, dstPixelType);
                        if (result != MvError.MV_OK)
                        {
                            Console.WriteLine("Image Convert failed:{0:x8}", result);
                            return;
                        }
                        Console.WriteLine("Image Convert success!");

                        string inputFilePath = string.Format("InputImage_w{0}_h{1}_{2}.raw", inputImage.Width, inputImage.Height, inputImage.PixelType);
                        string outputFilePath = string.Format("OutputImage_w{0}_h{1}_{2}.raw", outImage.Width, outImage.Height, outImage.PixelType);

                        //ch: 保存图像到文件 | en: Save image to file
                        using (FileStream fs = new FileStream(inputFilePath, FileMode.Create))
                        {
                            fs.Write(inputImage.PixelData, 0, inputImage.PixelData.Length);
                        }
                        Console.WriteLine("Save inputImage: {0}!", inputFilePath);

                        using (FileStream fs = new FileStream(outputFilePath, FileMode.Create))
                        {
                            fs.Write(outImage.PixelData, 0, outImage.PixelData.Length);
                        }
                        Console.WriteLine("Save OutputImage: {0}!", outputFilePath);

                        //ch: 图像使用完及时释放，防止内存快速上涨导致频繁GC |en：Release image promptly to prevent rapid memory increase leading to frequent GC.
                        outImage.Dispose();
                    }

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

            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
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
