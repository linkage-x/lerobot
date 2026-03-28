#ifndef MULTILIGHTCTRLDEMO_H
#define MULTILIGHTCTRLDEMO_H

#include <QMainWindow>
#include "MvCamera.h"
#include <map>
#include <QMap>
#include <QDebug>

#define DISPLAY_NUM 4  // 显示窗口个数
namespace Ui {
class MultiLightCtrlDemo;
}

class MultiLightCtrlDemo : public QMainWindow
{
    Q_OBJECT

public:
    explicit MultiLightCtrlDemo(QWidget *parent = 0);
    ~MultiLightCtrlDemo();

    void static __stdcall ImageCallBack2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree);
    void ImageCallBackInner2(MV_FRAME_OUT* pstFrame, bool bAutoFree);

private slots:
    void on_ContinusRadioButton_clicked();

    void on_TriggerRadioButton_clicked();

    void on_StartButton_clicked();
  
    void on_StopButton_clicked();

    void on_SoftwareCheckBox_clicked();

    void on_SoftwareButton_clicked();

    void on_EnumDeviceButton_clicked();

    void on_OpenButton_clicked();

    void on_CloseButton_clicked();

    void on_MultiLightcomboBox_currentTextChanged(const QString &arg1);

 
	void on_txtUserInput_textChanged(const QString &arg1);
	
    void on_chkUserInput_toggled(bool checked);

    void on_chkCamNode_toggled(bool checked);


private:
    void ShowErrorMsg(QString csMessage, unsigned int nErrorNum); // ch:显示错误信息窗口 | en: Show the window of error message
    void EnableControls(bool bIsCameraReady);// ch:判断按钮使能 | en:Enable the controls
    int SetTriggerMode();                    // ch:设置触发模式 | en:Set Trigger Mode
    int GetTriggerMode();                    // ch:获取触发模式 | en:Get Trigger Mode
    int GetTriggerSource();                  // ch:获取触发 | en:Get Trigger Source
    int GetMultiLight();                     // ch:获取多灯控制  | en:Get MultiLight
    int EXPOSURE_NUM;
    int UpdateCmbMultiLightInfo();
    unsigned char* pImageBufferList[DISPLAY_NUM];
    unsigned int   nImageBufferSize;
    MV_RECONSTRUCT_IMAGE_PARAM stImgReconstructionParam;

private:
    Ui::MultiLightCtrlDemo *ui;
    MV_CC_DEVICE_INFO_LIST  m_stDevList;                        // ch:设备信息链表 | en:The list of device info
    void*                   m_hWnd[DISPLAY_NUM];                // ch:显示窗口句柄 | en:The Handle of Display Window
    bool                    m_bOpenDevice;                      // ch:是否打开设备 | en:Whether to open device
    bool                    m_bStartGrabbing;                   // ch:是否开始抓图| en:Whether to start grabbing
    bool                    m_bSoftWareTriggerCheck;			// ch:是否软触发| en:Whether to software
    bool                    m_bContinueRadioButton;				// ch:连续模式按钮 | en:continue button
    bool                    m_bTriggerRadioButton;				// ch:触发模式按钮 | en:Trigger button


    CMvCamera*              m_pcMyCamera;                       // ch:相机类设备实例| en:The instance of CMvCamera
    int                     m_nTriggerMode;                     // ch:触发模式 | en:Trigger Mode
    int                     m_nTriggerSource;                   // ch:触发源   | en:Trigger Source

    unsigned char*          m_pBufForHB;                        // ch:HB解码缓冲区| en:The buffer of HB decode
    unsigned int            m_nBufSizeForHB;                    // ch:HB解码缓冲区长度| en:The length of HB decode buffer
    bool                    m_bHBFalg;                          // ch:当前图像是否为HB模式 | en:Cur pixelformat is HB mode
    bool                    m_bSupportMultiLightControl;       // ch:是否支持分时曝光节点 | en:Whether support MultiLightControl node
    bool                    m_bMultiNode;                      // 当前显示节点是不是多组曝光节点
    QMap<QString, int> m_mapMultiLight;
};

#endif // MULTILIGHTCTRLDEMO_H
