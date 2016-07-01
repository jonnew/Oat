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

#include "../../lib/datatypes/Sample.h"

#include "FrameServer.h"

namespace oat {

namespace pg = FlyCapture2;

class PGGigECam : public FrameServer {

public:
    PGGigECam(const std::string &frame_sink_address,
              const size_t index,
              const double fps);

    ~PGGigECam();

    // Use a configuration file to specify parameters
    void configure(void) override; // Default options
    void configure(const std::string &config_file,
                   const std::string &config_key) override;
    void connectToNode(void) override;
    bool serveFrame(void) override;
    void fireSoftwareTrigger(void);

private:

    // Timing stuff
    bool enforce_fps_ {false};
    static constexpr uint64_t IEEE_1394_HZ = {8000};
    uint64_t ieee_1394_cycle_index_ {0};
    uint64_t ieee_1394_start_cycle_ {0};
    bool ieee_1394_start_set_ {false};
    int last_ieee_1394_sec_ {0};
    bool first_frame_ {true};

    // Used to mark times between acquisitions
    oat::Sample::Microseconds tick_, tock_;

    // GigE Camera configuration
    unsigned int num_cameras_;
    int64_t max_index_ {0};
    size_t index_;

    int x_bin_ {1};
    int y_bin_ {1};
    float gain_db_ {0};
    float shutter_ms_ {0};
    float exposure_EV_ {0};
    bool acquisition_started_ {false};
    bool use_trigger_ {false};
    bool use_software_trigger_ {false};
    bool trigger_polarity_ {true};
    int64_t trigger_mode_ {14};
    int64_t trigger_source_pin_ {0};
    int64_t white_bal_red_ {0};
    int64_t white_bal_blue_ {0};
    double frames_per_second_ {30.0};
    //bool use_frame_buffer_ {false};
    //unsigned int num_transmit_retries_ {0};
    int64_t strobe_output_pin_ {1};

    // GigE Camera interface
    pg::GigECamera camera_;

    // The current, unbuffered frame in PG's format
    pg::Image raw_image_;
    std::unique_ptr<pg::Image> rgb_image_;

    // For establishing connection
    int setCameraIndex(unsigned int requested_idx);
    int connectToCamera(void);

    // Acquisition options
    // NOTE: These functions operate on member variables, and therefore the
    // arguments are gratuitous. However, these functions are by definition
    // not used in performance-critical sections of code, and I think decent
    // type sigs are a good trade for the extra copy operations. 
    int setupStreamChannels(void);
    int setupFrameRate(double fps, bool is_auto);
    int setupShutter(float shutter_ms);
    int setupShutter(bool is_auto);
    int setupGain(float gain_db);
    int setupGain(bool is_auto);
    int setupExposure(float exposure_EV);
    int setupExposure(bool is_auto);
    int setupWhiteBalance(int white_bal_red, int white_bal_blue);
    int setupWhiteBalance(bool is_on);
    int setupPixelBinning(int x_bin, int y_bin);
    int setupImageFormat(void);
    int setupDefaultImageFormat(void);
    //int setupCameraFrameBuffer(void);
    //TODO: int setupImageFormat(int xOffset, int yOffset, int height, int width, PixelFormat format);
    int setupTrigger(void);
    int setupEmbeddedImageData(void);

    // IEEE 1394 shutter open timestamp uncycling
    uint64_t uncycle1394Timestamp(int ieee_1394_sec,
                                  int ieee_1394_cycle);

    // Physical camera_ control
    int turnCameraOn(void);
    int grabImage(void);

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
