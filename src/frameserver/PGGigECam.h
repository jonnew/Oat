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

#include "FrameServer.h"

#include <memory>
#include <string>
#include <vector>

#include "FlyCapture2.h"
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Sample.h"

namespace oat {

namespace pg = FlyCapture2;

class PGGigECam : public FrameServer {

using rte = std::runtime_error;

public:
    /**
     * @brief Serve frames from a Point Grey GigE camera.
     * @param sink_address frame sink address
     */
    explicit PGGigECam(const std::string &sink_address);
    ~PGGigECam() final;

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

    void connectToNode(void) override;
    bool process(void) override;

private:

    // Timing stuff
    bool enforce_fps_ {false};
    double frames_per_second_ {30.0};
    static constexpr uint64_t IEEE_1394_HZ = {8000};
    uint64_t ieee_1394_cycle_index_ {0};
    uint64_t ieee_1394_start_cycle_ {0};
    bool ieee_1394_start_set_ {false};
    int last_ieee_1394_sec_ {0};
    bool first_frame_ {true};

    // Used to mark times between acquisitions
    oat::Sample::Microseconds tick_, tock_;

    bool acquisition_started_ {false};
    bool use_trigger_ {false};

    // GigE Camera object
    pg::GigECamera camera_;

    // The current, unbuffered frame in PG's format
    pg::Image raw_image_;
    std::unique_ptr<pg::Image> rgb_image_;

    // Acquisition settings routines 
    void setupStreamChannels(void);
    void setupFrameRate(double fps, bool is_auto = false);
    void setupShutter(float shutter_ms, bool is_auto = false);
    void setupGain(float gain_db, bool is_auto = false);
    void setupWhiteBalance(int bal_red, int bal_blue, bool is_on);
    void setupPixelBinning(size_t x_bin, size_t y_bin);
    void setupImageFormat(const std::vector<size_t> &roi);
    void setupImageFormat(void);
    //TODO: int setupCameraFrameBuffer(void);
    //TODO: int setupImageFormat(int xOffset, int yOffset, int height, int width, PixelFormat format);
    void setupAsyncTrigger(int trigger_mode, bool trigger_rising, int trigger_pin);
    void setupStrobeOutput(int strobe_pin);
    void setupEmbeddedImageData(void);
    void setupGrabSettings(void);

    // IEEE 1394 shutter open timestamp uncycling
    uint64_t uncycle1394Timestamp(int ieee_1394_sec,
                                  int ieee_1394_cycle);

    // Physical camera control
    void turnCameraOn(void);
    void connectToCamera(size_t index);
    void startCapture(void);

    /** 
     * @brief Grab a frame from the camera's buffer.
     * 
     * @return return code meaning:
     *   -1 : Grab timeout occurred
     *    0 : Successful grab
     *  >=1 : A return code greater than or equal to 1  only occurs if a strict
     *        frame rate is enforced when using an external trigger. Because it is
     *        possible for PG cameras to skip triggers, the function will check the
     *        time between consecutive frames and indicate the estimated number of
     *        missed frames, if any, using this code.
     */
    int grabImage(void);

    // Diagnostics and meta
    size_t findNumCameras(void);
    void printError(pg::Error error);
    bool pollForTriggerReady(void);
    void printCameraInfo(void);
    void printStreamChannelInfo(pg::GigEStreamChannel *stream_channel);
};

}      /* namespace oat */
#endif /* OAT_PGGIGECAM_H */
