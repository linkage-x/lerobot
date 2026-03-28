#include "multilightctrldemo.h"
#include "ui_multilightctrldemo.h"
#include <stdio.h>
#include <unistd.h>
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QStyleFactory>
#include <QTextCodec>

MultiLightCtrlDemo::MultiLightCtrlDemo(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MultiLightCtrlDemo)
{
    ui->setupUi(this);
    m_pcMyCamera = NULL;
    m_bOpenDevice = false;
    m_bStartGrabbing = false;
    m_bContinueRadioButton = false;
    m_bSoftWareTriggerCheck = false;
    m_bTriggerRadioButton = false;
    m_nTriggerMode = MV_TRIGGER_MODE_OFF;
    m_nTriggerSource = MV_TRIGGER_SOURCE_SOFTWARE;
    EXPOSURE_NUM=0;
    memset(&m_stDevList, 0 , sizeof(m_stDevList));
    m_hWnd[0] = (void*)ui->Display1widget->winId();
    m_hWnd[1] = (void*)ui->Display2widget->winId();
    m_hWnd[2] = (void*)ui->Display3widget->winId();
    m_hWnd[3] = (void*)ui->Display4widget->winId();
    ui->MultiLightcomboBox->setEnabled(false);
    EnableControls(false);

    ui->txtUserInput->setEnabled(false);
	ui->chkUserInput->setEnabled(false);
	ui->chkCamNode->setEnabled(false);
   
   for(int i=0;i<DISPLAY_NUM;i++)
    {
        pImageBufferList[i]=NULL;
    }
    nImageBufferSize = 0;
    memset(&stImgReconstructionParam,0,sizeof(MV_RECONSTRUCT_IMAGE_PARAM));

    m_pBufForHB = NULL;
    m_nBufSizeForHB = 0;
    m_bHBFalg = false;
    m_bSupportMultiLightControl = false;
    m_bMultiNode = false;
}

MultiLightCtrlDemo::~MultiLightCtrlDemo()
{
    for(int i=0;i<DISPLAY_NUM;i++)
    {
        if(NULL!=pImageBufferList[i])
        {
            free(pImageBufferList[i]);
            pImageBufferList[i]=NULL;
        }
    }
    nImageBufferSize = 0;

    if(m_pBufForHB != NULL)
    {
        free(m_pBufForHB);
        m_pBufForHB = NULL;
    }
    m_nBufSizeForHB = 0;

    delete ui;
}

void MultiLightCtrlDemo::on_ContinusRadioButton_clicked()
{
    if(true == ui->ContinusRadioButton->isChecked())
    {
        m_pcMyCamera->SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF);
        m_bContinueRadioButton = true;
        m_bTriggerRadioButton = false;

    }
    EnableControls(true);
}

void MultiLightCtrlDemo::on_TriggerRadioButton_clicked()
{
    if(true == ui->TriggerRadioButton->isChecked())
    {
        m_pcMyCamera->SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON);
        m_bTriggerRadioButton = true;
        m_bContinueRadioButton = false;

    }
    EnableControls(true);
}

void MultiLightCtrlDemo::on_StartButton_clicked()
{
    ui->Display1widget->setEnabled(false);
    ui->Display1widget->setEnabled(true);
    ui->Display2widget->setEnabled(false);
    ui->Display2widget->setEnabled(true);
    ui->Display3widget->setEnabled(false);
    ui->Display3widget->setEnabled(true);
    ui->Display4widget->setEnabled(false);
    ui->Display4widget->setEnabled(true);

    ui->MultiLightcomboBox->setEnabled(false);
    ui->txtUserInput->setEnabled(false);

    if(EXPOSURE_NUM <=0)
    {

         if(ui->chkUserInput->isChecked())
         {
              EXPOSURE_NUM = 1;
              ui->txtUserInput->setText("1");
         }
         
    }


    m_pcMyCamera->RegisterImageCallBack2(ImageCallBack2, this, true);

    int nRet = MV_OK;
    MVCC_INTVALUE_EX stIntInfo = {0};
    nRet = m_pcMyCamera->GetIntValue("PayloadSize", &stIntInfo);
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Get PayloadSize fail!", nRet);
        return;
    }
    m_nBufSizeForHB = stIntInfo.nCurValue;

    MVCC_ENUMVALUE stParam = {0};
    nRet = m_pcMyCamera->GetEnumValue("ImageCompressionMode", &stParam);
    if (MV_OK != nRet)
    {
        m_bHBFalg = false;
    }
    else
    {
        if (0 == stParam.nCurValue) 
        {
            m_bHBFalg = false;
        }
        else if(2 == stParam.nCurValue)
        {
            m_bHBFalg = true;
            if(m_pBufForHB != NULL)
            {
                free(m_pBufForHB);
                m_pBufForHB = NULL;
            }

            m_pBufForHB = (unsigned char*)malloc(m_nBufSizeForHB);
            if(NULL == m_pBufForHB)
            {
                ShowErrorMsg("malloc HB uffer fail!", MV_E_RESOURCE);
                return;
            }
            EXPOSURE_NUM = EXPOSURE_NUM & 0xF;
        }
        else 
        {
            m_bHBFalg = false;
        }
    }

    nRet = m_pcMyCamera->StartGrabbing();
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Start grabbing fail", nRet);
        return;
    }
    m_bStartGrabbing = true;
    EnableControls(true);
    ui->chkCamNode->setEnabled(false);
    ui->chkUserInput->setEnabled(false);
    printf("light num=%d\n",EXPOSURE_NUM);

}

void MultiLightCtrlDemo::on_StopButton_clicked()
{
    if (false == m_bOpenDevice || false == m_bStartGrabbing || NULL == m_pcMyCamera)
    {
        return;
    }
    int nRet = m_pcMyCamera->StopGrabbing();
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Stop grabbing fail", nRet);
        return;
    }
    m_bStartGrabbing = false;
    EnableControls(true);

    if(ui->chkUserInput->isChecked())
    {
        ui->chkUserInput->setEnabled(true);
    }

    if(ui->chkCamNode->isChecked())
    {
         ui->chkCamNode->setEnabled(true);
    }

    if((false == ui->chkUserInput->isChecked()) &&(false == ui->chkCamNode->isChecked()))
    {
        ui->chkCamNode->setEnabled(true);
        ui->chkUserInput->setEnabled(true);
    }

    int nCount = ui->MultiLightcomboBox->count();
    if(0 == nCount)
    {
         ui->chkUserInput->setEnabled(true);
         ui->chkCamNode->setEnabled(false);
    }


    ui->txtUserInput->setEnabled((m_bOpenDevice&& ui->chkUserInput->isChecked()) ? true : false);
    ui->MultiLightcomboBox->setEnabled((m_bOpenDevice  && ui->chkCamNode->isChecked()) ? true : false);

}

void MultiLightCtrlDemo::on_SoftwareCheckBox_clicked()
{
    if(true == ui->SoftwareCheckBox->isChecked())
    {
        m_bSoftWareTriggerCheck = true;
        int nRet = m_pcMyCamera->SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
        if (MV_OK != nRet)
        {
            ShowErrorMsg("Set Trigger Source SoftWare fail", nRet);
            return;
        }
    }
    else
    {
        m_bSoftWareTriggerCheck = false;
    }
    EnableControls(true);
}

void MultiLightCtrlDemo::on_SoftwareButton_clicked()
{
    if (true != m_bStartGrabbing)
    {
        return;
    }

    int nRet = m_pcMyCamera->CommandExecute("TriggerSoftware");
    if (nRet != MV_OK)
    {
        ShowErrorMsg("Trigger Software fail", nRet);
        return;
    }
    EnableControls(true);
}

void MultiLightCtrlDemo::on_EnumDeviceButton_clicked()
{
    ui->DevicecomboBox->clear();
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("GBK"));
    ui->DevicecomboBox->setStyle(QStyleFactory::create("Windows"));
    memset(&m_stDevList, 0, sizeof(m_stDevList));

    int nRet = CMvCamera::EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
                                      MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE ,&m_stDevList);
    if (MV_OK != nRet)
    {
        return;
    }

    for (unsigned int i = 0; i < m_stDevList.nDeviceNum; i++)
    {

        MV_CC_DEVICE_INFO* pDeviceInfo = m_stDevList.pDeviceInfo[i];
        if (NULL == pDeviceInfo)
        {
            continue;
        }
        char strUserName[256] = {0};
        if (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE)
        {
            int nIp1 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            int nIp2 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            int nIp3 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            int nIp4 = (m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stGigEInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]GigE:   %s (%s) (%d.%d.%d.%d)", i, pDeviceInfo->SpecialInfo.stGigEInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stGigEInfo.chSerialNumber, nIp1, nIp2, nIp3, nIp4);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]GigE:   %s (%s) (%d.%d.%d.%d)", i, pDeviceInfo->SpecialInfo.stGigEInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stGigEInfo.chSerialNumber, nIp1, nIp2, nIp3, nIp4);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_USB_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]UsbV3:  %s (%s)", i, pDeviceInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]UsbV3:  %s (%s)", i, pDeviceInfo->SpecialInfo.stUsb3VInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_CAMERALINK_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stCMLInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]CML:  %s (%s)", i,pDeviceInfo->SpecialInfo.stCMLInfo.chUserDefinedName,
                    pDeviceInfo->SpecialInfo.stCMLInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]CML:  %s (%s)", i,pDeviceInfo->SpecialInfo.stCMLInfo.chModelName,
                    pDeviceInfo->SpecialInfo.stCMLInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stCXPInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]CXP:  %s (%s)", i,pDeviceInfo->SpecialInfo.stCXPInfo.chUserDefinedName,
                    pDeviceInfo->SpecialInfo.stCXPInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]CXP:  %s (%s)", i,pDeviceInfo->SpecialInfo.stCXPInfo.chModelName,
                    pDeviceInfo->SpecialInfo.stCXPInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_XOF_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stXoFInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]XOF:  %s (%s)", i,pDeviceInfo->SpecialInfo.stXoFInfo.chUserDefinedName,
                    pDeviceInfo->SpecialInfo.stXoFInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]XOF:  %s (%s)",i, pDeviceInfo->SpecialInfo.stXoFInfo.chModelName,
                    pDeviceInfo->SpecialInfo.stXoFInfo.chSerialNumber);
            }
        }
        else
        {
            ShowErrorMsg("Unknown device enumerated", 0);
        }
        ui->DevicecomboBox->addItem(QString::fromLocal8Bit(strUserName));
    }

    if (0 == m_stDevList.nDeviceNum)
    {
        ShowErrorMsg("No device", 0);
        return;
    }
    ui->DevicecomboBox->setCurrentIndex(0);
    EnableControls(true);

}

void MultiLightCtrlDemo::on_OpenButton_clicked()
{
    if (true == m_bOpenDevice || NULL != m_pcMyCamera)
    {
        return;
    }
    int nIndex = ui->DevicecomboBox->currentIndex();
    if ((nIndex < 0) | (nIndex >= MV_MAX_DEVICE_NUM))
    {
        ShowErrorMsg("Please select device", 0);
        return;
    }

    if (NULL == m_stDevList.pDeviceInfo[nIndex])
    {
        ShowErrorMsg("Device does not exist", 0);
        return;
    }

    if(m_pcMyCamera == NULL)
    {
        m_pcMyCamera = new CMvCamera;
        if (NULL == m_pcMyCamera)
        {
            return;
        }
    }

    int nRet = m_pcMyCamera->Open(m_stDevList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        delete m_pcMyCamera;
        m_pcMyCamera = NULL;
        ShowErrorMsg("Open Fail", nRet);
        return;
    }

    if (m_stDevList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
    {
        unsigned int nPacketSize = 0;
        nRet = m_pcMyCamera->GetOptimalPacketSize(&nPacketSize);
        if (nRet == MV_OK)
        {
            nRet = m_pcMyCamera->SetIntValue("GevSCPSPacketSize",nPacketSize);
            if(nRet != MV_OK)
            {
                ShowErrorMsg("Warning: Set Packet Size fail!", nRet);
            }
        }
        else
        {
            ShowErrorMsg("Warning: Get Packet Size fail!", nRet);
        }
    }

    nRet = GetMultiLight();
    if (nRet != MV_OK)
    {
        m_bSupportMultiLightControl = false;
        ui->chkUserInput->setEnabled(true);
        ui->chkUserInput->setChecked(true);

        ui->txtUserInput->setEnabled(true);

        ui->chkCamNode->setEnabled(false);
        ui->chkCamNode->setChecked(false);

       EXPOSURE_NUM = ui->txtUserInput->text().toInt();

       ShowErrorMsg("Get MultiLight Fail", nRet);
    }
    else
    {
        m_bSupportMultiLightControl = true;

        ui->chkCamNode->setEnabled(true);
        ui->chkCamNode->setChecked(true);

        ui->chkUserInput->setEnabled(false);
        ui->chkUserInput->setChecked(false);

        // 判断是不是多重曝光 是的话控件使能
        MVCC_ENUMVALUE stEnumValue = {0};
        nRet = m_pcMyCamera->GetEnumValue("MultiLightControl", &stEnumValue);
        if(MV_OK == nRet)
        {
            if (0x12 == stEnumValue.nCurValue)  // 说明是多组曝光
            {
                m_bMultiNode = true;
            }
            else
            {
                m_bMultiNode = false;
            }
        }
    }


    m_bOpenDevice = true;

    nRet = GetTriggerMode();
    if (nRet != MV_OK)
    {
        ShowErrorMsg("Get Trigger Mode Fail", nRet);
    }

    nRet = GetTriggerSource();
    if (nRet != MV_OK)
    {
        ShowErrorMsg("Get Trigger Source Fail", nRet);
    }

    EnableControls(true);

}

void MultiLightCtrlDemo::on_CloseButton_clicked()
{
    if(true == m_bStartGrabbing)
    {
        int nRet = m_pcMyCamera->StopGrabbing();
        if (MV_OK != nRet)
        {
            ShowErrorMsg("Stop grabbing fail", nRet);
            return;
        }
        m_bStartGrabbing = false;
    }

    if (m_pcMyCamera)
    {
        m_pcMyCamera->Close();
        delete m_pcMyCamera;
        m_pcMyCamera = NULL;
    }

    m_bStartGrabbing = false;
    m_bOpenDevice = false;
    EnableControls(true);

    ui->txtUserInput->clear();
    ui->MultiLightcomboBox->clear();
	
    ui->MultiLightcomboBox->setEnabled(false);

    ui->chkUserInput->setChecked(false);
    ui->chkUserInput->setEnabled(false);
    ui->txtUserInput->setEnabled(false);


    ui->chkCamNode->setChecked(false);
    ui->chkCamNode->setEnabled(false);


}

// ch:显示错误信息 | en:Show error message
void MultiLightCtrlDemo::ShowErrorMsg(QString csMessage, unsigned int nErrorNum)
{
    QString errorMsg = csMessage;
    if (nErrorNum != 0)
    {
        QString TempMsg;
        TempMsg.sprintf(": Error = %x: ", nErrorNum);
        errorMsg += TempMsg;
    }

    switch(nErrorNum)
    {
    case MV_E_HANDLE:           errorMsg += "Error or invalid handle ";                                         break;
    case MV_E_SUPPORT:          errorMsg += "Not supported function ";                                          break;
    case MV_E_BUFOVER:          errorMsg += "Cache is full ";                                                   break;
    case MV_E_CALLORDER:        errorMsg += "Function calling order error ";                                    break;
    case MV_E_PARAMETER:        errorMsg += "Incorrect parameter ";                                             break;
    case MV_E_RESOURCE:         errorMsg += "Applying resource failed ";                                        break;
    case MV_E_NODATA:           errorMsg += "No data ";                                                         break;
    case MV_E_PRECONDITION:     errorMsg += "Precondition error, or running environment changed ";              break;
    case MV_E_VERSION:          errorMsg += "Version mismatches ";                                              break;
    case MV_E_NOENOUGH_BUF:     errorMsg += "Insufficient memory ";                                             break;
    case MV_E_ABNORMAL_IMAGE:   errorMsg += "Abnormal image, maybe incomplete image because of lost packet ";   break;
    case MV_E_UNKNOW:           errorMsg += "Unknown error ";                                                   break;
    case MV_E_GC_GENERIC:       errorMsg += "General error ";                                                   break;
    case MV_E_GC_ACCESS:        errorMsg += "Node accessing condition error ";                                  break;
    case MV_E_ACCESS_DENIED:	errorMsg += "No permission ";                                                   break;
    case MV_E_BUSY:             errorMsg += "Device is busy, or network disconnected ";                         break;
    case MV_E_NETER:            errorMsg += "Network error ";                                                    break;

    }

    QMessageBox::information(NULL, "PROMPT", errorMsg);
}

void MultiLightCtrlDemo::EnableControls(bool bIsCameraReady)
{
    ui->OpenButton->setEnabled(m_bOpenDevice ? false : (bIsCameraReady ? true : false));
    ui->CloseButton->setEnabled((m_bOpenDevice && bIsCameraReady) ? true : false);
    ui->StartButton->setEnabled((m_bStartGrabbing && bIsCameraReady) ? false : (m_bOpenDevice ? true : false));
    ui->StopButton->setEnabled(m_bStartGrabbing ? true : false);
    ui->SoftwareCheckBox->setEnabled((m_bOpenDevice ) ? true : false);
    ui->SoftwareButton->setEnabled((m_bStartGrabbing && m_bSoftWareTriggerCheck && m_bTriggerRadioButton) ? true : false);
    ui->ContinusRadioButton->setEnabled(m_bOpenDevice ? true : false);
    ui->TriggerRadioButton->setEnabled(m_bOpenDevice ? true : false);
    ui->SoftwareCheckBox->setChecked(m_bSoftWareTriggerCheck);


    ui->chkCamNode->setEnabled(m_bOpenDevice && !m_bStartGrabbing && m_bSupportMultiLightControl ? true : false);
    ui->MultiLightcomboBox->setEnabled((m_bOpenDevice && !m_bStartGrabbing && ui->chkCamNode->isChecked() ) ? true : false);

    ui->chkUserInput->setEnabled(m_bOpenDevice && !m_bStartGrabbing && (false == m_bSupportMultiLightControl) ? true : false);
    ui->txtUserInput->setEnabled((m_bOpenDevice && !m_bStartGrabbing && ui->chkUserInput->isChecked() ) ? true : false);
}

void __stdcall MultiLightCtrlDemo::ImageCallBack2(MV_FRAME_OUT* pstFrame, void *pUser, bool bAutoFree)
{
    if(NULL != pUser)
    {
        MultiLightCtrlDemo *pMainWindow = (MultiLightCtrlDemo*)pUser;
        pMainWindow->ImageCallBackInner2(pstFrame, bAutoFree);
    }
}

void MultiLightCtrlDemo::ImageCallBackInner2(MV_FRAME_OUT* pstFrame, bool bAutoFree)
{
    int nRet = MV_OK;
    MV_CC_IMAGE stDisplayInfo;
    memset(&stDisplayInfo, 0, sizeof(MV_CC_IMAGE));
    MV_CC_HB_DECODE_PARAM stHBParam;
    memset(&stHBParam, 0, sizeof(MV_CC_HB_DECODE_PARAM));
    
    
    do 
    {
                 // if HB FLAG，first Decode 
        if(true == m_bHBFalg)
        {
            stHBParam.pSrcBuf = pstFrame->pBufAddr;
            stHBParam.nSrcLen = pstFrame->stFrameInfo.nFrameLen;
            stHBParam.pDstBuf = m_pBufForHB;
            stHBParam.nDstBufSize = m_nBufSizeForHB;
            nRet = m_pcMyCamera->HB_Decode(&stHBParam);
            if (MV_OK != nRet)
            {
                ShowErrorMsg("HB Decode failed", nRet);
                if(NULL != m_pBufForHB)
                {
                    free(m_pBufForHB);
                    m_pBufForHB = NULL;
                }
                m_nBufSizeForHB = 0;
                break;
            }
        }

        //no need to reconstruct image
        if(pstFrame->stFrameInfo.nSubImageNum > 0)
        {
            if(pstFrame->stFrameInfo.nSubImageNum <= EXPOSURE_NUM)
            {
                for (unsigned int i=0;i<pstFrame->stFrameInfo.nSubImageNum;i++)
                {
                    stDisplayInfo.pImageBuf = pstFrame->stFrameInfo.SubImageList.pstSubImage[i].pImageBuf;
                    stDisplayInfo.nImageLen = pstFrame->stFrameInfo.SubImageList.pstSubImage[i].nImageLen;
                    stDisplayInfo.nWidth = pstFrame->stFrameInfo.SubImageList.pstSubImage[i].nWidth;
                    stDisplayInfo.nHeight = pstFrame->stFrameInfo.SubImageList.pstSubImage[i].nHeight;
                    stDisplayInfo.enPixelType = pstFrame->stFrameInfo.SubImageList.pstSubImage[i].enPixelType;
                    nRet = m_pcMyCamera->DisplayOneFrame(m_hWnd[i],&stDisplayInfo);
                    if (MV_OK != nRet)
                    {
                        printf("subimage[%d]:DisplayOneFrame failed, nRet=%x\n", i, nRet);
                    }
                }
            }
            else
            {
                printf("ERROR:sub image number[%d] is larger than light number[%d]",pstFrame->stFrameInfo.nSubImageNum,EXPOSURE_NUM);
                break;
            }
        }
        else //need to reconstruct image
        {
            if (EXPOSURE_NUM > 1)      // 多灯
            {
                stImgReconstructionParam.nWidth = pstFrame->stFrameInfo.nExtendWidth;
                stImgReconstructionParam.nHeight = pstFrame->stFrameInfo.nExtendHeight;
                stImgReconstructionParam.enPixelType = pstFrame->stFrameInfo.enPixelType;
                stImgReconstructionParam.pSrcData = pstFrame->pBufAddr;
                stImgReconstructionParam.nSrcDataLen = pstFrame->stFrameInfo.nFrameLenEx;
                stImgReconstructionParam.nExposureNum = EXPOSURE_NUM;
                stImgReconstructionParam.enReconstructMethod = MV_SPLIT_BY_LINE;

                if(true == m_bHBFalg)
                {
                    stImgReconstructionParam.nWidth = stHBParam.nWidth;
                    stImgReconstructionParam.nHeight = stHBParam.nHeight;
                    stImgReconstructionParam.enPixelType = stHBParam.enDstPixelType;
                    stImgReconstructionParam.pSrcData = stHBParam.pDstBuf;
                    stImgReconstructionParam.nSrcDataLen = stHBParam.nDstBufLen;
                    stImgReconstructionParam.nExposureNum = EXPOSURE_NUM;
                    stImgReconstructionParam.enReconstructMethod = MV_SPLIT_BY_LINE;
                }

                if (stImgReconstructionParam.nSrcDataLen / (EXPOSURE_NUM-1) > nImageBufferSize)
                {
                    for (unsigned int i = 0; i < DISPLAY_NUM; i++)
                    {
                        if (pImageBufferList[i])
                        {
                            free(pImageBufferList[i]);
                            pImageBufferList[i] = NULL;
                        }

                        pImageBufferList[i] = (unsigned char*)malloc(sizeof(unsigned char) * stImgReconstructionParam.nSrcDataLen / (EXPOSURE_NUM-1));
                        if (NULL != pImageBufferList[i])
                        {
                            stImgReconstructionParam.stDstBufList[i].pBuf = pImageBufferList[i];
                            stImgReconstructionParam.stDstBufList[i].nBufSize = stImgReconstructionParam.nSrcDataLen/ (EXPOSURE_NUM-1);
                        }
                        else
                        {
                            printf("malloc failed, index[%d], size[%d] \n",i, stImgReconstructionParam.nSrcDataLen/ (EXPOSURE_NUM-1));
                            break;
                        }
                    }

                    nImageBufferSize = stImgReconstructionParam.nSrcDataLen / (EXPOSURE_NUM-1);
                }

                // Split Image And Reconstruct Image
                nRet = m_pcMyCamera->ReconstructImage(&stImgReconstructionParam);
                if (MV_OK != nRet)
                {
                    printf("ReconstructImage failed 0x%x", nRet);
                    break;
                }

                for (unsigned int i = 0; i < EXPOSURE_NUM; i++)
                {
                    stDisplayInfo.pImageBuf = stImgReconstructionParam.stDstBufList[i].pBuf;
                    stDisplayInfo.nImageLen = stImgReconstructionParam.stDstBufList[i].nBufLen;
                    stDisplayInfo.nWidth = stImgReconstructionParam.stDstBufList[i].nWidth;
                    stDisplayInfo.nHeight = stImgReconstructionParam.stDstBufList[i].nHeight;
                    stDisplayInfo.enPixelType = stImgReconstructionParam.stDstBufList[i].enPixelType;

                    nRet = m_pcMyCamera->DisplayOneFrame(m_hWnd[i],&stDisplayInfo);
                    if (MV_OK != nRet)
                    {
                        printf("index[%d]: DisplayOneFrame failed, nRet=%x\n",i,nRet);
                        break;
                    }
                }
            }
            else
            {
                if(true == m_bHBFalg)
                {
                    stDisplayInfo.pImageBuf = stHBParam.pDstBuf;
                    stDisplayInfo.nImageLen = stHBParam.nDstBufLen;
                    stDisplayInfo.nWidth = stHBParam.nWidth;
                    stDisplayInfo.nHeight = stHBParam.nHeight;
                    stDisplayInfo.enPixelType = stHBParam.enDstPixelType;
                }
                else
                {
                    stDisplayInfo.pImageBuf = pstFrame->pBufAddr;
                    stDisplayInfo.nImageLen = pstFrame->stFrameInfo.nFrameLenEx;
                    stDisplayInfo.nWidth = pstFrame->stFrameInfo.nExtendWidth;
                    stDisplayInfo.nHeight = pstFrame->stFrameInfo.nExtendHeight;
                    stDisplayInfo.enPixelType = pstFrame->stFrameInfo.enPixelType;
                }
                nRet = m_pcMyCamera->DisplayOneFrame(m_hWnd[0], &stDisplayInfo);
                if (MV_OK != nRet)
                {
                    printf("DisplayOneFrame failed, nRet=%x\n",nRet);
                    break;
                }
            }
        }
        
    }while(0);
    
    if (false == bAutoFree) //非自动释放模式，需要手动释放资源
    {
        m_pcMyCamera->FreeImageBuffer(pstFrame);
    }
}

int MultiLightCtrlDemo::SetTriggerMode()
{
    return m_pcMyCamera->SetEnumValue("TriggerMode", m_nTriggerMode);
}

int MultiLightCtrlDemo::GetTriggerMode()
{
    MVCC_ENUMVALUE stEnumValue = {0};

    int nRet = m_pcMyCamera->GetEnumValue("TriggerMode", &stEnumValue);
    if (MV_OK != nRet)
    {
       return nRet;
    }
    m_nTriggerMode = stEnumValue.nCurValue;
    if (MV_TRIGGER_MODE_ON ==  m_nTriggerMode)
    {
        m_bTriggerRadioButton = true;
        ui->TriggerRadioButton->setChecked(true);
    }
    else
    {
        m_nTriggerMode = MV_TRIGGER_MODE_OFF;
        m_bTriggerRadioButton = false;
        ui->ContinusRadioButton->setChecked(true);
    }
    EnableControls(true);
    return MV_OK;
}
int MultiLightCtrlDemo::UpdateCmbMultiLightInfo()
{

    MVCC_ENUMVALUE stMultiLightControlValue = { 0 };

    int nRet = m_pcMyCamera->GetEnumValue("MultiLightControl", &stMultiLightControlValue);
    if (MV_OK != nRet)
    {
        return nRet;
    }

    for (int i = 0; i < stMultiLightControlValue.nSupportedNum; i++)
    {
        if (stMultiLightControlValue.nCurValue == stMultiLightControlValue.nSupportValue[i])
        {
            ui->MultiLightcomboBox->setCurrentIndex(i);
        }
    }

    return MV_OK;
}
int MultiLightCtrlDemo::GetMultiLight()
{
    m_mapMultiLight.clear();
    MVCC_ENUMVALUE stMultiLightControlValue = { 0 };
    MVCC_ENUMENTRY stMultiLightControlEntry = { 0 };

    int nRet = m_pcMyCamera->GetEnumValue("MultiLightControl", &stMultiLightControlValue);
    if (MV_OK != nRet)
    {
        return nRet;
    }

    EXPOSURE_NUM= stMultiLightControlValue.nCurValue&0xF;

    ui->MultiLightcomboBox->clear();
    for (int i = 0; i < stMultiLightControlValue.nSupportedNum; i++)
    {
        memset(&stMultiLightControlEntry, 0, sizeof(stMultiLightControlEntry));
        stMultiLightControlEntry.nValue = stMultiLightControlValue.nSupportValue[i];
        m_pcMyCamera->GetEnumEntrySymbolic("MultiLightControl", &stMultiLightControlEntry);

        ui->MultiLightcomboBox->addItem((QString)stMultiLightControlEntry.chSymbolic);

        m_mapMultiLight.insert((QString)stMultiLightControlEntry.chSymbolic, stMultiLightControlEntry.nValue);
    }

    for (int i = 0; i < stMultiLightControlValue.nSupportedNum; i++)
    {
        if (stMultiLightControlValue.nCurValue == stMultiLightControlValue.nSupportValue[i])
        {
            ui->MultiLightcomboBox->setCurrentIndex(i);
        }
    }

    return MV_OK;
}

int MultiLightCtrlDemo::GetTriggerSource()
{
    MVCC_ENUMVALUE stEnumValue = {0};

    int nRet = m_pcMyCamera->GetEnumValue("TriggerSource", &stEnumValue);
    if (MV_OK != nRet)
    {
        return nRet;
    }

    if ((unsigned int)MV_TRIGGER_SOURCE_SOFTWARE == stEnumValue.nCurValue)
    {
        m_bSoftWareTriggerCheck = true;
    }
    else
    {
        m_bSoftWareTriggerCheck = false;
    }

    return MV_OK;
}

void MultiLightCtrlDemo::on_MultiLightcomboBox_currentTextChanged(const QString &arg1)
{
    if(false == m_bOpenDevice)
    {
        return;
    }
    for (QMap<QString, int>::iterator it = m_mapMultiLight.begin(); it != m_mapMultiLight.end(); it++)
    {
        if (it.key() == arg1)
        {
            int nRet = m_pcMyCamera->SetEnumValue("MultiLightControl", it.value());
            if (MV_OK != nRet)
            {
                ShowErrorMsg("Set MultiLightControl fail", nRet);
                return;
            }
            EXPOSURE_NUM=it.value();
            break;
        }
    }

    MVCC_ENUMVALUE stEnumValue = {0};
    int nRet = m_pcMyCamera->GetEnumValue("MultiLightControl", &stEnumValue);
    if (MV_OK != nRet)
    {
          ShowErrorMsg("Get Multi Light Control failed, Not Support!", nRet);
          return ;
     }
     EXPOSURE_NUM = stEnumValue.nCurValue & 0x0F;
}

void MultiLightCtrlDemo::on_txtUserInput_textChanged(const QString &arg1)
{
	if(false == m_bOpenDevice)
	{
		return;
	}
    QIntValidator *validator = new QIntValidator(ui->txtUserInput);

    validator->setRange(1, 4);

    ui->txtUserInput->setValidator(validator);

    int nUserInput =  ui->txtUserInput->text().toInt();


    if(m_pcMyCamera)
    {
        int nRet = m_pcMyCamera->SetEnumValue("MultiLightControl", nUserInput);
        if(MV_OK == nRet)
        {
            // 同步修改界面控件显示
            GetMultiLight();
            UpdateCmbMultiLightInfo();
        }

    }
    EXPOSURE_NUM = nUserInput;
}

void MultiLightCtrlDemo::on_chkUserInput_toggled(bool checked)
{
    if( m_bOpenDevice)
    {
        if (ui->chkUserInput->isChecked() )
         {
             ui->chkCamNode->setChecked(false);
             ui->chkCamNode->setEnabled(false);
             ui->txtUserInput->setEnabled(true);
             ui->MultiLightcomboBox->setEnabled(false);

             EXPOSURE_NUM = ui->txtUserInput->text().toInt();
         }
         else
         {
            if(0 == ui->MultiLightcomboBox->count())
            {
                ui->chkUserInput->setEnabled(true);
                ui->chkCamNode->setEnabled(false);
                ui->txtUserInput->setEnabled(false);
            }
            else
            {
                ui->chkCamNode->setEnabled(true);
                ui->txtUserInput->setEnabled(false);
            }

         }
    }

}

void MultiLightCtrlDemo::on_chkCamNode_toggled(bool checked)
{
     if( m_bOpenDevice)
     {
         if (ui->chkCamNode->isChecked())
         {
             ui->chkUserInput->setChecked(false);
             ui->chkUserInput->setEnabled(false);
             ui->txtUserInput->setEnabled(false);
             ui->MultiLightcomboBox->setEnabled(true);
             // 若没有切换combox 获取当前值
             MVCC_ENUMVALUE stEnumValue = {0};
             int nRet = m_pcMyCamera->GetEnumValue("MultiLightControl", &stEnumValue);
             if(MV_OK != nRet)
             {
                 return ;
             }

             EXPOSURE_NUM = stEnumValue.nCurValue&0xF;
         }
         else
         {
             ui->chkUserInput->setEnabled(true);
             ui->MultiLightcomboBox->setEnabled(false);
         }
     }
}


