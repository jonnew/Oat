//******************************************************************************
//* File:   PGGigECam.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef OAT_PGGIGECAM_H
#define OAT_PGGIGECAM_H

#include <memory>
#include <string>
#include <opencv2/core/mat.hpp>

#include "FlyCapture2.h"

#include "FrameServer.h"

namespace oat {

namespace pg = FlyCapture2;

class PGGigECam : public FrameServer {

public:
    PGGigECam(const std::string &frame_sink_address, 
              const size_t index,
              const double fps);

    // Use a configuration file to specify parameters
    void configure(void) override; // Default options
    void configure(const std::string &config_file,
                   const std::string &config_key) override;
    void connectToNode(void) override;
    bool serveFrame(void) override;
    void fireSoftwareTrigger(void);

private:

    unsigned int num_cameras;

    // GigE Camera configuration
    static constexpr int64_t min_index {0};
    int64_t max_index {0};
    size_t index_;

    int x_bin {1};
    int y_bin {1};
    float gain_dB {0};
    float shutter_ms {0};
    float exposure_EV {0};
    bool aquisition_started {false};
    bool use_trigger {false};
    bool use_software_trigger {false};
    bool trigger_polarity {true};
    int64_t trigger_mode {14};
    int64_t trigger_source_pin {0};
    int64_t white_bal_red {0};
    int64_t white_bal_blue {0};
    double frames_per_second {30.0};
    bool use_camera_frame_buffer {false};
    unsigned int number_transmit_retries {0};

    // GigE Camera interface
    pg::GigECamera camera;

    // Camera and control state info
    pg::CameraInfo camera_info;
    pg::TriggerModeInfo trigger_mode_info;
    pg::GigEImageSettingsInfo image_settings_info;

    // The current, unbuffered frame in PG's format
    pg::Image raw_image;
    std::unique_ptr<pg::Image> rgb_image;

    // For establishing connection
    int setCameraIndex(unsigned int requested_idx);
    int connectToCamera(void);

    // Acquisition options
    int setupStreamChannels(void);
    int setupFrameRate(double fps, bool is_auto);
    int setupShutter(float shutter_ms);
    int setupShutter(bool is_auto);
    int setupGain(float gain_dB);
    int setupGain(bool is_auto);
    int setupExposure(float exposure_EV);
    int setupExposure(bool is_auto);
    int setupWhiteBalance(int white_bal_red, int white_bal_blue);
    int setupWhiteBalance(bool is_on);
    int setupPixelBinning(void);
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

    // Diagnostics and meta
    int findNumCameras(void);
    void printError(pg::Error error);
    bool pollForTriggerReady(void);
    int printCameraInfo(void);
    int printBusInfo(void);
    void printStreamChannelInfo(pg::GigEStreamChannel *stream_channel);

    // TODO: Grabbed frame callback
    // void onGrabbedImage(FlyCapture2::Image* pImage, const void* pCallbackData);
};

}      /* namespace oat */
#endif /* OAT_PGGIGECAM_H */
