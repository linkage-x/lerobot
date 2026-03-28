/*
 * 这个示例演示了录像功能。
 * This sample shows the how to recording.
 * 
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;
using System.Threading;

namespace Image_Recording
{
    class Program
    {
        const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
            | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        static volatile bool _grabThreadExit = false;
        static void FrameGrabThread(object obj)
        {
            int ret = MvError.MV_OK;

            IDevice device = obj as IDevice;
            IVideoRecorder recorder = device.VideoRecorder;
            IStreamGrabber streamGrabber = device.StreamGrabber;

            while (!_grabThreadExit)
            {
                IFrameOut frame;

                //ch：获取一帧图像 | en: Get one frame
                ret = streamGrabber.GetImageBuffer(1000, out frame);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                    continue;
                }

                Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                //ch：图像添加到录像文件 | en: Record the frame
                ret = recorder.InputOneFrame(frame.Image);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Input one frame failed:{0:x8}", ret);
                }

                //ch: 释放图像缓存  | en: Release the image buffer
                streamGrabber.FreeImageBuffer(frame);
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

                // ch:设置触发模式为off | en:set trigger mode as off
                ret = device.Parameters.SetEnumValue("TriggerMode", 0);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                device.StreamGrabber.SetImageNodeNum(5);

                // ch:开启抓图 | en: start grab image
                ret = device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                // ch:获取录像所需的参数 | en: Get record parameters
                IIntValue widthValue;
                IIntValue heightValue;
                IEnumValue pixelTypeValue;
                IFloatValue frameRateValue;
                device.Parameters.GetIntValue("Width", out widthValue);
                device.Parameters.GetIntValue("Height", out heightValue);
                device.Parameters.GetEnumValue("PixelFormat", out pixelTypeValue);
                device.Parameters.GetFloatValue("ResultingFrameRate", out frameRateValue);

                // ch:开启录像 | en: start record
                RecordParam recordParam = new RecordParam();
                recordParam.Width = (uint)widthValue.CurValue;
                recordParam.Height = (uint)heightValue.CurValue;
                recordParam.PixelType = (MvGvspPixelType)pixelTypeValue.CurEnumEntry.Value;

                // ch:帧率(大于1/16)fps | en:Frame Rate (>1/16)fps
                recordParam.FrameRate = frameRateValue.CurValue;
                // ch:码率kbps(128kbps-16Mbps) | en:Bitrate kbps(128kbps-16Mbps)
                recordParam.BitRate = 1000;
                // ch:录像格式(仅支持AVI) | en:Record Format(AVI is only supported)
                recordParam.FormatType = VideoFormatType.AVI;

                ret = device.VideoRecorder.StartRecord("./Recording.avi", recordParam);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start record failed:{0:x8}", ret);
                    return;
                }

                // ch:开启抓图线程 | en: Start the grabbing thread
                Thread GrabThread = new Thread(FrameGrabThread);
                GrabThread.Start(device);

                Console.WriteLine("Press enter to exit");
                Console.ReadLine();

                //ch: 通知线程退出 | en: Notify the grab thread to exit
                _grabThreadExit = true;
                GrabThread.Join();

                // ch:停止录像 | en:Stop record
                ret = device.VideoRecorder.StopRecord();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Stop record failed:{0:x8}", ret);
                }

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
