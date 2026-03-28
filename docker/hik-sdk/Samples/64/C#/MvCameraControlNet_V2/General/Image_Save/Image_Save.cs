/*
 * 这个示例演示了如何保存图像数据
 * This sample demonstrates how to save image data
 */

using MvCameraControl;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace Image_Save
{
    class Image_Save
    {
        const DeviceTLayerType enumTLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLGigEDevice
    | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLCameraLinkDevice | DeviceTLayerType.MvGenTLXoFDevice;
        public void Run()
        {
            IDevice device = null;
            List<IDeviceInfo> deviceInfos;

            try
            {
                int result = DeviceEnumerator.EnumDevices(enumTLayerType, out deviceInfos);
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

                // ch:选择要保存的文件类型 | en: Select the file type for image save
                Console.WriteLine("Please select the file type to save:");
                Console.WriteLine("0: Raw");
                Console.WriteLine("1: " + ImageFormatType.Bmp);
                Console.WriteLine("2: " + ImageFormatType.Jpeg);
                Console.WriteLine("3: " + ImageFormatType.Png);
                Console.WriteLine("4: " + ImageFormatType.Tiff);
                int imageFormatType = Convert.ToInt32(Console.ReadLine());
                if (imageFormatType < 0 || imageFormatType > 4)
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
                    int packetSize;
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

                    // ch:如果图像是HB格式，需要先解码 | en:If the image is HB format, should to be decoded first
                    IFrameOut frameForSave = frameOut;
                    if (frameOut.Image.PixelType.ToString().Contains("HB"))
                    {
                        result = device.ImageDecoder.HBDecode(frameOut, out frameForSave);
                        if (result != MvError.MV_OK)
                        {
                            Console.WriteLine("HB Decode failed:{0:x8}", result);
                            return;
                        }
                    }

                    // ch:保存图像 | en:Save image
                    string fileName = string.Format("Image_w{0}_h{1}_p{2}", frameForSave.Image.Width, frameForSave.Image.Height, frameForSave.Image.PixelType);
                    switch(imageFormatType)
                    {
                        case 0:
                            {
                                // ch:保存raw数据 | en:Save raw data
                                fileName += ".raw";
                                using (FileStream fs = new FileStream(fileName, FileMode.Create))
                                {
                                    // ch:图像超过int.MaxValue时, 分段保存 | en:The save image in segments When byte array is large than int.MaxValue
                                    UInt64 intMaxValue = (UInt64)int.MaxValue;
                                    if (frameForSave.Image.ImageSize > intMaxValue)
                                    {
                                        int blockLen = 1024 * 1024;
                                        byte[] newData = new byte[blockLen];

                                        UInt64 remain = frameForSave.Image.ImageSize;

                                        IntPtr ptrSourceTemp = frameForSave.Image.PixelDataPtr;
                                        for (UInt64 i = 0; i < remain; i++)
                                        {
                                            int writeLen = (int)(remain > (UInt64)blockLen ? (UInt64)blockLen : remain);
                                            Marshal.Copy(ptrSourceTemp, newData, 0, writeLen); 
                                            fs.Write(newData, 0, writeLen);

                                            remain -= (UInt64)writeLen;
                                            ptrSourceTemp = new IntPtr(ptrSourceTemp.ToInt64() + writeLen);
                                        }
                                    }
                                    else
                                    {
                                        fs.Write(frameForSave.Image.PixelData, 0, (int)frameForSave.Image.ImageSize);
                                    }
                                }
                                Console.WriteLine("Save image success! " + fileName);
                            }
                            break;
                        case 1:
                        case 2:
                        case 3:
                        case 4:
                            {
                                ImageFormatInfo imageFormatInfo = new ImageFormatInfo();
                                imageFormatInfo.FormatType = (ImageFormatType)imageFormatType;

                                // ch:JPEG格式需要配置图像质量 | en:Set JpegQuality for JPEG file
                                if (imageFormatInfo.FormatType == ImageFormatType.Jpeg)
                                {
                                    imageFormatInfo.JpegQuality = 80;
                                }

                                // ch:图像保存的文件名 | en:Save image to file
                                fileName += "." + imageFormatInfo.FormatType.ToString();

                                // ch:保存图像 | en:Save image to file 
                                result = device.ImageSaver.SaveImageToFile(fileName, frameForSave.Image, imageFormatInfo, CFAMethod.Equilibrated);
                                if (result != MvError.MV_OK)
                                {
                                    Console.WriteLine("SaveImageToFile failed:{0:x8}", result);
                                    return;
                                }
                                Console.WriteLine("Save image success! " + fileName);
                            }
                            break;
                        default:
                            Console.WriteLine("Input file type error");
                            return;
                    }

                    //ch: 图像使用完及时释放，防止内存快速上涨导致频繁GC | en：Release image promptly to prevent rapid memory increase leading to frequent GC.
                    if (frameForSave != frameOut)
                    {
                        frameForSave.Image.Dispose();
                    }
                }

                //ch: 释放图像缓存 | en: Release image buffer
                device.StreamGrabber.FreeImageBuffer(frameOut);

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
            }
        }
        static void Main(string[] args)
        {
            //ch: 初始化SDK |  en: Initialize SDK
            SDKSystem.Initialize();

            Image_Save program = new Image_Save();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
