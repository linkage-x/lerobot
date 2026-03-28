#-------------------------------------------------
#
# Project created by QtCreator 2024-01-23T22:12:59
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TRANSLATIONS = MultiLightCtrlDemo_zh_EN.ts

TARGET = MultiLightCtrlDemo
TEMPLATE = app


SOURCES += main.cpp\
        multilightctrldemo.cpp \
    MvCamera.cpp

HEADERS  += multilightctrldemo.h \
    MvCamera.h

FORMS    += multilightctrldemo.ui

INCLUDEPATH += $$PWD/../../../../../../include/

LIBS += -L$$PWD/../../../../../..//bin -L$$PWD/../../../../../../lib/32/ -lMvCameraControl

RESOURCES += \
    resource.qrc
