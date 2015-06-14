//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#ifndef CAMERACONTROL_H
#define CAMERACONTROL_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "FlyCapture2.h"

#include "Camera.h"

class PGGigECam : public Camera {
public:
    PGGigECam(std::string frame_sink_name);

    // Use a configuration file to specify parameters
    void configure(void); // Default options
    void configure(std::string config_file, std::string key);

    void grabMat(void);
    bool serveMat(void);
    void fireSoftwareTrigger(void);

private:

    // Camera name
    std::string name;

    // Size and offset of the image to aquire
    cv::Size frame_size, frame_offset;

    unsigned int num_cameras, index;
    float gain_dB, shutter_ms, exposure_EV;
    bool aquisition_started;
    bool use_trigger;
    bool use_software_trigger;
    int trigger_polarity, trigger_mode, trigger_source_pin;
    int white_bal_red, white_bal_blue;
    double frames_per_second;
    bool use_camera_frame_buffer;
    unsigned int number_transmit_retries;
    
    // GigE Camera interface
    FlyCapture2::GigECamera camera;

    // Camera and control state info
    FlyCapture2::CameraInfo camera_info;
    FlyCapture2::TriggerModeInfo trigger_mode_info;
    FlyCapture2::GigEImageSettingsInfo image_settings_info;

    // The current, unbuffered frame in PG's format
    FlyCapture2::Image raw_image;
    FlyCapture2::Image rgb_image;

    // For establishing connection
    int setCameraIndex(unsigned int requested_idx);
    int connectToCamera(void);

    // Acquisition options 
    int setupStreamChannels(void);
    int setupShutter(float shutter_ms);
    int setupShutter(bool is_auto);
    int setupGain(float gain_dB);
    int setupGain(bool is_auto);
    int setupExposure(float exposure_EV);
    int setupExposure(bool is_auto);
    int setupWhiteBalance(int white_bal_red, int white_bal_blue);
    int setupWhiteBalance(bool is_on);
    int setupImageFormat(void);
    int setupDefaultImageFormat(void);
    int setupCameraFrameBuffer(void);
    //TODO: int setupImageFormat(int xOffset, int yOffset, int height, int width, PixelFormat format);
    //int setupImageBinning(int xBinFactor, int yBinFactor);
    int setupTrigger(void);

    // Physical camera control
    int turnCameraOn(void);
    //TODO: int turnCameraOff(void);
    void grabImage(void);

    // Convert flycap image to cv::Mat
    cv::Mat imageToMat(void);

    int findNumCameras(void);
    void printError(FlyCapture2::Error error);
    bool pollForTriggerReady(void);
    int printCameraInfo(void);
    int printBusInfo(void);
    void printStreamChannelInfo(FlyCapture2::GigEStreamChannel *stream_channel);
    
    // TODO: Grabbed frame callback
    // void onGrabbedImage(FlyCapture2::Image* pImage, const void* pCallbackData);


};

#endif //CAMERACONFIG_H
