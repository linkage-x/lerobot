/*
* 这个示例演示了线阵相机采用2组分时曝光参数(需要相机先配置好分时曝光功能)，先拆图再自上而下拼图，并将拼好的裸图以bmp格式保存至本地。
* This sample shows the linear array camera uses 2 sets of time-sharing exposure to acquire raw data
* then disassemble the raw data and  top-down puzzle
* and last save to bmp format to local.
*/

using MvCameraControl;
using System;
using System.Collections.Generic;

namespace MultiLightCtrl_ImageStitching
{
    class MultiLightCtrl_ImageStitching
    {
        /// <summary>
        /// ch:传输层类型 | en: Transport layer type
        /// </summary>
        const DeviceTLayerType enumTLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice| DeviceTLayerType.MvGenTLGigEDevice | DeviceTLayerType.MvGenTLCXPDevice 
            | DeviceTLayerType.MvGenTLCameraLinkDevice | DeviceTLayerType.MvGenTLXoFDevice;

        public void Run()
        {
            int result = MvError.MV_OK;
            IDevice device = null;
            List<IDeviceInfo> deviceInfos;

            uint exposureNum = 2; // 分时频闪的灯数, 默认曝光个数为2

            try
            {
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

                // ch:图像的曝光个数 || en:Exposure num
                Console.Write("Please input exposure num:");
                exposureNum = Convert.ToUInt32(Console.ReadLine());
                if (exposureNum == 0)
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
                    int packetSize = 0;

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
                result = device.StreamGrabber.GetImageBuffer(20000, out frameOut);
                if (MvError.MV_OK == result)
                {
                    Console.WriteLine("Get Image Buffer: Width[{0}] , Height[{1}] , FrameNum[{2}]"
                        , frameOut.Image.Width, frameOut.Image.Height, frameOut.FrameNum);

                    // ch:如果图像是HB格式，需要先解码 | en:If the image is HB format, should to be decoded first
                    IFrameOut frameRaw = frameOut;
                    if (frameOut.Image.PixelType.ToString().Contains("HB"))
                    {
                        result = device.ImageDecoder.HBDecode(frameOut, out frameRaw);
                        if (MvError.MV_OK != result)
                        {
                            Console.WriteLine("HB Decode failed:{0:x8}", result);
                            return;
                        }
                    }

                    ImageReconstructionMethod imageReconstructionMethod = ImageReconstructionMethod.SplitByLine;

                    // ch:图像重构并拼接 | en:Image Reconstruct and Stitching
                    IImage outImage;
                    result = device.ImageProcess.ReconstructImage(frameRaw.Image, exposureNum, imageReconstructionMethod, ImageStitchingMethod.Vertical, out outImage);
                    if (result != MvError.MV_OK)
                    {
                        Console.WriteLine("Reconstruct Image failed:{0:x8}", result);
                        return;
                    }
                    Console.WriteLine("Reconstruct Image success!");

                    string resultImageFilePath = "result.bmp";
                    ImageFormatInfo imageFormatInfo = new ImageFormatInfo();
                    imageFormatInfo.FormatType = ImageFormatType.Bmp;
                    result = device.ImageSaver.SaveImageToFile(resultImageFilePath, outImage, imageFormatInfo, CFAMethod.Equilibrated);
                    if (result != MvError.MV_OK)
                    {
                        Console.WriteLine("Save Image failed:{0:x8}", result);
                        return;
                    }
                    Console.WriteLine("Save Image success! {0}", resultImageFilePath);

                    //ch: 图像使用完及时释放，防止内存快速上涨导致频繁GC | en：Release image promptly to prevent rapid memory increase leading to frequent GC.
                    if (frameRaw != frameOut)
                    {
                        frameRaw.Image.Dispose();
                    }

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
            }
        }
        static void Main(string[] args)
        {
            //ch: 初始化SDK |  en: Initialize SDK
            SDKSystem.Initialize();

            MultiLightCtrl_ImageStitching program = new MultiLightCtrl_ImageStitching();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
