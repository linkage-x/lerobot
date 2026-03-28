/*
 * 这个示例演示如何使用GigE相机的ActionCommand触发功能。
 * This program shows how to use the ActionCommand trigger function of a GigE camera.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MvCameraControl;
namespace Grab_ActionCommand
{
    class Grab_ActionCommand
    {
        static bool _bExit = false;
        static uint _deviceKey = 1;
        static uint _groupKey = 1;
        static uint _groupMask = 1;

        public static void ActionCommandWorkThread()
        {          
            int ret = MvError.MV_OK;
            ActionCmdInfo actionCmdInfo = new ActionCmdInfo();
            List<ActionCmdResult> actionCmdResults;

            actionCmdInfo.DeviceKey = _deviceKey;
            actionCmdInfo.GroupKey = _groupKey;
            actionCmdInfo.GroupMask = _groupMask;
            actionCmdInfo.BroadcastAddress = "255.255.255.255";
            actionCmdInfo.TimeOut = 100;
            actionCmdInfo.ActionTimeEnable = 0;

            while(!_bExit)
            {
                ret = DeviceEnumerator.GigEIssueActionCommand(actionCmdInfo, out actionCmdResults);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Issue Action Command failed! nRet {0:x8}", ret);
                    continue;
                }

                if (actionCmdResults != null)
                {
                    for (int i = 0; i < actionCmdResults.Count; i++)
                    {
                        Console.WriteLine("Ip == {0}, Status == {1}", actionCmdResults[i].DeviceAddress, actionCmdResults[i].Status);
                    }
                }
            }
           
        }

        public static void ReceiveImageWorkThread(object obj)
        {
            IStreamGrabber streamGrabber = (IStreamGrabber)obj;

            while (true)
            {
                IFrameOut frame;

                //ch：获取一帧图像 | en: Get one frame
                int ret = streamGrabber.GetImageBuffer(1000, out frame);
                if (ret == MvError.MV_OK)
                {
                    Console.WriteLine("Get one frame: Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                    //ch: 释放图像缓存  | en: Release the image buffer
                    streamGrabber.FreeImageBuffer(frame);                  
                }
                else
                {
                    Console.WriteLine("Get Image failed:{0:x8}", ret);
                }

                if (_bExit)
                {
                    break;
                }
            }

        }

        static void Main(string[] args)
        {
            int ret = MvError.MV_OK;

            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();
            IDevice device = null;

            try
            {
                List<IDeviceInfo> devInfoList;

                // ch:枚举设备 | en:Enum device
                ret = DeviceEnumerator.EnumDevices(DeviceTLayerType.MvGigEDevice, out devInfoList);
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
                   
                    if (devInfo.TLayerType == DeviceTLayerType.MvGigEDevice || devInfo.TLayerType == DeviceTLayerType.MvVirGigEDevice || devInfo.TLayerType == DeviceTLayerType.MvGenTLGigEDevice)
                    {
                        IGigEDeviceInfo gigeDevInfo = devInfo as IGigEDeviceInfo;
                        uint nIp1 = ((gigeDevInfo.CurrentIp & 0xff000000) >> 24);
                        uint nIp2 = ((gigeDevInfo.CurrentIp & 0x00ff0000) >> 16);
                        uint nIp3 = ((gigeDevInfo.CurrentIp & 0x0000ff00) >> 8);
                        uint nIp4 = (gigeDevInfo.CurrentIp & 0x000000ff);
                        Console.WriteLine("[Device {0}]:", devIndex);
                        Console.WriteLine("DevIP: {0}.{1}.{2}.{3}", nIp1, nIp2, nIp3, nIp4);
                        Console.WriteLine("ModelName: {0}\n",devInfo.ModelName);
                    }
                    else
                    {
                        Console.Write("Not Support!\n");
                        break;
                    }
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

                // ch:打开触发模式 || en:set trigger mode as On
                ret = device.Parameters.SetEnumValue("TriggerMode", 1);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

                // ch:设置触发源为Action1 | en:Set trigger source as Action1
                ret = device.Parameters.SetEnumValueByString("TriggerSource", "Action1");
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Trigger Source failed! {0:x8}", ret);
                    return;
                }

                // ch:设置Action Device Key | en:Set Action Device Key
                ret = device.Parameters.SetIntValue("ActionDeviceKey", _deviceKey);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Action Device Key failed! {0:x8}", ret);
                    return;
                }

                // ch:设置Action Group Key | en:Set Action Group Key
                ret = device.Parameters.SetIntValue("ActionGroupKey", _groupKey);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Action Group Key failed! {0:x8}", ret);
                    return;
                }

                // ch:设置Action Group Mask | en:Set Action Group Mask
                ret = device.Parameters.SetIntValue("ActionGroupMask", _groupMask);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set Action Group Mask fail! {0:x8}", ret);
                    return;
                }

                // ch:开启抓图 || en: start grab image
                ret = device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                Thread hActionCommandThreadHandle = new Thread(ActionCommandWorkThread);
                hActionCommandThreadHandle.Start();

                Thread hReceiveImageThreadHandle = new Thread(ReceiveImageWorkThread);
                hReceiveImageThreadHandle.Start(device.StreamGrabber);

                Console.WriteLine("Press enter to exit");
                Console.ReadKey();

                _bExit = true;
                hActionCommandThreadHandle.Join();
                hReceiveImageThreadHandle.Join();

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
