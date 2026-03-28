/*
 * 这个示例演示了取图时深拷贝图像，将图像放入队列后异步处理。
 * This sample shows how to perform a deep copy of the image during image acquisition, place the image in a queue, and process it asynchronously.
 * 
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MvCameraControl;
using System.Threading;

namespace Grab_ImageClone
{
    class Grab_ImageClone
    {
        private const DeviceTLayerType devLayerType = DeviceTLayerType.MvGigEDevice | DeviceTLayerType.MvUsbDevice | DeviceTLayerType.MvGenTLCameraLinkDevice
            | DeviceTLayerType.MvGenTLCXPDevice | DeviceTLayerType.MvGenTLXoFDevice;

        /// <summary>
        /// ch: 帧缓存队列 | en: frame queue for process
        /// </summary>
        private Queue<IFrameOut> _frameQueue = null;

        /// <summary>
        /// ch: 队列图像数量上限 | en: maximum number of frames in the queue
        /// </summary>
        private const uint _maxQueueSize = 10; 

        /// <summary>
        /// ch: 异步处理线程 | asynchronous processing thread
        /// </summary>
        private Thread _asyncProcessThread = null;
        /// <summary>
        /// ch: 信号，通知异步处理线程处理 | Used to notify the processing thread
        /// </summary>
        private Semaphore _frameGrabSem = null;
        /// <summary>
        /// ch: 异步处理线程退出标志 | en: Flag to notify the  processing thread to exit
        /// </summary>
        private volatile bool _processThreadExit = false;

        public Grab_ImageClone()
        {
            _frameQueue = new Queue<IFrameOut>();
            _frameGrabSem = new Semaphore(0, Int32.MaxValue);
        }


        public void Run()
        {
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

                // ch:设置触发模式为off || en:set trigger mode as off
                ret = device.Parameters.SetEnumValue("TriggerMode", 0);
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Set TriggerMode failed:{0:x8}", ret);
                    return;
                }

                //ch: 设置合适的缓存节点数量 | en: Setting the appropriate number of image nodes
                device.StreamGrabber.SetImageNodeNum(5);

                //ch：创建异步处理线程 | en: Create an asynchronous processing thread
                _processThreadExit = false;
                _asyncProcessThread = new Thread(AsyncProcessThread);
                _asyncProcessThread.Start();

                // ch:注册回调函数 | en:Register image callback
                device.StreamGrabber.FrameGrabedEvent += FrameGrabedEventHandler;
                // ch:开启抓图 || en: start grab image
                ret = device.StreamGrabber.StartGrabbing();
                if (ret != MvError.MV_OK)
                {
                    Console.WriteLine("Start grabbing failed:{0:x8}", ret);
                    return;
                }

                Console.WriteLine("Press enter to stop grabbing");
                Console.ReadLine();

                //ch: 通知异步处理线程退出 | en: Notify the thread to exit
                _processThreadExit = true;
                _asyncProcessThread.Join();

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

            }
        }

        void AsyncProcessThread()
        {
            try
            {
                while (!_processThreadExit)
                {
                    if (_frameGrabSem.WaitOne(100))
                    {
                        using (IFrameOut frame = _frameQueue.Dequeue())
                        {
                            Console.WriteLine("AsyncProcessThread: process one frame, Width[{0}] , Height[{1}] , FrameNum[{2}]", frame.Image.Width, frame.Image.Height, frame.FrameNum);

                            //Processing the image data, such as algorithms
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("AsyncProcessThread exception: " + e.Message);
            }

        }

        void FrameGrabedEventHandler(object sender, FrameGrabbedEventArgs e)
        {
            Console.WriteLine("FrameGrabedEventHandler: Get one frame, Width[{0}] , Height[{1}] , FrameNum[{2}]", e.FrameOut.Image.Width, e.FrameOut.Image.Height, e.FrameOut.FrameNum);

            try
            {
                
                lock (this)
                {
                    if (_frameQueue.Count <= _maxQueueSize)
                    {
                        // ch: 克隆图像数据（深拷贝） | en :Clone frame data using deep copy
                        IFrameOut frameCopy = (IFrameOut)e.FrameOut.Clone();

                        //ch: 添加到队列并通知处理线程 | en: Add the frame to the queue and notify the processing thread
                        _frameQueue.Enqueue(frameCopy);
                        _frameGrabSem.Release();
                    }
                    
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine("FrameGrabedEventHandler exception: " + exception.Message);
            }

        }

        static void Main(string[] args)
        {
            // ch: 初始化 SDK | en: Initialize SDK
            SDKSystem.Initialize();

            Grab_ImageClone program = new Grab_ImageClone();
            program.Run();

            Console.WriteLine("Press enter to exit");
            Console.ReadKey();

            // ch: 反初始化SDK | en: Finalize SDK
            SDKSystem.Finalize();
        }
    }
}
