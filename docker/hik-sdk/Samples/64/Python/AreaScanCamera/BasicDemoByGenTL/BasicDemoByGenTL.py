# -- coding: utf-8 --
import os

from PyQt5.QtWidgets import *
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *
from PyUICBasicDemoByGenTL import Ui_MainWindow


# 获取选取设备信息的索引，通过[]之间的字符去解析
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()


# 将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


if __name__ == "__main__":

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    global deviceList
    deviceList = MV_GENTL_DEV_INFO_LIST()

    global interfaceList
    interfaceList = MV_GENTL_IF_INFO_LIST()

    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global isOpen
    isOpen = False
    global isGrabbing
    isGrabbing = False
    global isCalibMode  # 是否是标定模式（获取原始图像）
    isCalibMode = True


    # 绑定下拉列表至设备信息索引
    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[", "]", ui.comboDevice.get())

    # Decoding Characters
    def decoding_char(ctypes_char_array):
        """
        安全地从 ctypes 字符数组中解码出字符串。
        适用于 Python 2.x 和 3.x，以及 32/64 位环境。
        """
        byte_str = memoryview(ctypes_char_array).tobytes()
        
        # 在第一个空字符处截断
        null_index = byte_str.find(b'\x00')
        if null_index != -1:
            byte_str = byte_str[:null_index]
        
        # 多编码尝试解码
        for encoding in ['gbk', 'utf-8', 'latin-1']:
            try:
                return byte_str.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用替换策略
        return byte_str.decode('latin-1', errors='replace')

    # ch:枚举采集卡 | en:enum interfaces
    def enum_interfaces():
        global interfaceList
        global obj_cam_operation

        # 对话框选择cti文件
        # 使用当前工作目录作为默认路径，支持跨平台
        current_dir = os.getcwd()
        fileName, fileType = QFileDialog.getOpenFileName(mainWindow, "选择cti文件",
                                                         directory=current_dir,
                                                         filter="Cti文件(*cti)")
        if fileName is None or len(fileName) == 0:
            return -1

        interfaceList = MV_GENTL_IF_INFO_LIST()
        ret = MvCamera.MV_CC_EnumInterfacesByGenTL(interfaceList, fileName)
        if ret != 0:
            strError = "Enum interfaces fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return ret

        if interfaceList.nInterfaceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no interface", QMessageBox.Ok)
            return ret
        print("Find %d interfaces!" % interfaceList.nInterfaceNum)

        ifListTemp = []
        for i in range(0, interfaceList.nInterfaceNum):
            ifInfoTemp = cast(interfaceList.pIFInfo[i], POINTER(MV_GENTL_IF_INFO)).contents
            chTLType = decoding_char(ifInfoTemp.chTLType)
            chInterfaceID = decoding_char(ifInfoTemp.chInterfaceID)
            chDisplayName = decoding_char(ifInfoTemp.chDisplayName)

            ifListTemp.append("[" + str(i) + "] " + chTLType + ": " + chInterfaceID + " " + chDisplayName)

        ui.comboInterface.clear()
        ui.comboInterface.addItems(ifListTemp)
        ui.comboInterface.setCurrentIndex(0)

    # ch:枚举相机 | en:enum devices
    def enum_devices():
        global deviceList
        global interfaceList
        global obj_cam_operation

        deviceList = MV_GENTL_DEV_INFO_LIST()
        nIFIndex = ui.comboInterface.currentIndex()
        ret = MvCamera.MV_CC_EnumDevicesByGenTL(interfaceList.pIFInfo[nIFIndex], deviceList)
        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return ret

        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print("Find %d devices!" % deviceList.nDeviceNum)

        devListTemp = []
        for i in range(0, deviceList.nDeviceNum):
            devInfoTemp = cast(deviceList.pDeviceInfo[i], POINTER(MV_GENTL_DEV_INFO)).contents

            chDeviceID = decoding_char(devInfoTemp.chDeviceID)
            devListTemp.append("[" + str(i) + "] " + chDeviceID)

        ui.comboDevice.clear()
        ui.comboDevice.addItems(devListTemp)
        ui.comboDevice.setCurrentIndex(0)

    # ch:打开相机 | en:open device
    def open_device():
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global isOpen
        if isOpen:
            QMessageBox.warning(mainWindow, "Error", 'Camera is Running!', QMessageBox.Ok)
            return MV_E_CALLORDER

        nSelCamIndex = ui.comboDevice.currentIndex()
        if nSelCamIndex < 0:
            QMessageBox.warning(mainWindow, "Error", 'Please select a camera!', QMessageBox.Ok)
            return MV_E_CALLORDER

        obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if 0 != ret:
            strError = "Open device failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            isOpen = False
        else:
            set_continue_mode()

            get_param()

            isOpen = True
            enable_controls()

    # ch:开始取流 | en:Start grab image
    def start_grabbing():
        global obj_cam_operation
        global isGrabbing

        ret = obj_cam_operation.Start_grabbing(ui.widgetDisplay.winId())
        if ret != 0:
            strError = "Start grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            isGrabbing = True
            enable_controls()

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing():
        global obj_cam_operation
        global isGrabbing
        ret = obj_cam_operation.Stop_grabbing()
        if ret != 0:
            strError = "Stop grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            isGrabbing = False
            enable_controls()

    # ch:关闭设备 | Close device
    def close_device():
        global isOpen
        global isGrabbing
        global obj_cam_operation

        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False

        isGrabbing = False

        enable_controls()

    # ch:设置触发模式 | en:set trigger mode
    def set_continue_mode():
        strError = None

        ret = obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is " + str(is_trigger_mode)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.radioContinueMode.setChecked(True)
            ui.radioTriggerMode.setChecked(False)
            ui.bnSoftwareTrigger.setEnabled(False)

    # 设置软触发模式
    def set_software_trigger_mode():

        ret = obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            strError = "Set trigger mode failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.radioContinueMode.setChecked(False)
            ui.radioTriggerMode.setChecked(True)
            ui.bnSoftwareTrigger.setEnabled(isGrabbing)

    # ch:设置触发命令 | en:set trigger software
    def trigger_once():
        ret = obj_cam_operation.Trigger_once()
        if ret != 0:
            strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

    # ch:存图
    def save_bmp():
        ret = obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            print("Save image success")

    # ch: 获取参数
    def get_param():
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
            ui.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
            ui.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

    # ch: 设置参数
    def set_param():
        frame_rate = ui.edtFrameRate.text()
        exposure = ui.edtExposureTime.text()
        gain = ui.edtGain.text()
        ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

        return MV_OK

    # 设置控件状态
    def enable_controls():
        global isGrabbing
        global isOpen

        # 先设置group的状态，再单独设置各控件状态
        ui.groupGrab.setEnabled(isOpen)
        ui.groupParam.setEnabled(isOpen)

        ui.bnOpen.setEnabled(not isOpen)
        ui.bnClose.setEnabled(isOpen)

        ui.bnStart.setEnabled(isOpen and (not isGrabbing))
        ui.bnStop.setEnabled(isOpen and isGrabbing)
        ui.bnSoftwareTrigger.setEnabled(isGrabbing and ui.radioTriggerMode.isChecked())

        ui.bnSaveImage.setEnabled(isOpen and isGrabbing)

    # 初始化app, 绑定控件与函数
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    ui.bnEnumInterface.clicked.connect(enum_interfaces)
    ui.bnEnumDevice.clicked.connect(enum_devices)
    ui.bnOpen.clicked.connect(open_device)
    ui.bnClose.clicked.connect(close_device)
    ui.bnStart.clicked.connect(start_grabbing)
    ui.bnStop.clicked.connect(stop_grabbing)

    ui.bnSoftwareTrigger.clicked.connect(trigger_once)
    ui.radioTriggerMode.clicked.connect(set_software_trigger_mode)
    ui.radioContinueMode.clicked.connect(set_continue_mode)

    ui.bnGetParam.clicked.connect(get_param)
    ui.bnSetParam.clicked.connect(set_param)
    ui.bnSaveImage.clicked.connect(save_bmp)

    # 重写窗口关闭事件
    def closeEvent(event):
        try:
            close_device()
        except Exception as e:
            print("close_device failed: ", e)
        event.accept()

    mainWindow.closeEvent = closeEvent

    mainWindow.show()

    # PyQt5兼容性检查
    try:
        if hasattr(app, 'exec_'):
            app.exec_()
        else:
            app.exec()
    except Exception as e:
        print("Application execution failed: ", e)

    # 等待一段时间确保设备完全关闭
    import time
    time.sleep(0.2)


    MvCamera.MV_CC_Finalize()


    sys.exit()
