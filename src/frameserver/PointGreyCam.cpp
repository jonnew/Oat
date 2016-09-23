//******************************************************************************
//* File:   PointGreyCam.cpp
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

#include "PointGreyCam.h"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "FlyCapture2.h"
#include <cpptoml.h>
#include <opencv2/opencv.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

// Initialize Pixel map
template <typename T>
const typename PointGreyCam<T>::PixelMap PointGreyCam<T>::pix_map_ =
{
    {PIX_GREY,
        std::make_tuple(pg::PIXEL_FORMAT_MONO8, pg::PIXEL_FORMAT_MONO8, CV_8UC1)},
    {PIX_BGR,
        std::make_tuple(pg::PIXEL_FORMAT_RAW8, pg::PIXEL_FORMAT_BGR, CV_8UC3)}
};

template <typename T>
PointGreyCam<T>::PointGreyCam(const std::string &sink_address)
: FrameServer(sink_address)
{
    // Initialize frame timing
    tick_ = oat::Sample::Microseconds(0);
    tock_ = oat::Sample::Microseconds(0);
}

template <typename T>
PointGreyCam<T>::~PointGreyCam()
{
    // Ignore error return values -- throwing exception unsafe in destructor
    camera_.StopCapture();
    camera_.Disconnect();
}

template <typename T>
void PointGreyCam<T>::appendOptions(po::options_description &opts)
{
    // Accepts default options
    FrameServer::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("index,i", po::value<int>(),
         "Camera index. Defaults to 0. Useful in multi-camera imaging "
         "configurations.")
        ("fps,r", po::value<double>(),
         "Acquisition frame rate in Hz. Ignored if trigger-mode > -1 and "
         "enforce_fps=false. Defaults to the maximum frame rate.")
        ("enforce-fps,e",
         "If true, ensures that frames are produced at the fps setting bool "
         "retransmitting frames if the requested period is exceeded. This is "
         "sometimes needed in the case of an external trigger because PG cameras "
         "sometimes just ignore them. I have opened a support ticket on this, "
         "but PG has no solution yet.")
        ("shutter,s", po::value<double>(),
         "Shutter time in milliseconds. Defaults to auto.")
        ("color,C", po::value<std::string>(),
         "Pixel color format. Defaults to BRG.\n"
         "Values:\n"
         "  GREY: \t 8-bit Greyscale image.\n"
         "  BRG: \t8-bit, 3-chanel, BGR Color image.\n")
        ("gain,g", po::value<double>(),
         "Sensor gain value, specified in dB. Defaults to auto.")
        ("strobe-pin,S", po::value<size_t>(),
         "Hardware pin number on that a gate signal for the camera shutter "
         "is copied to. Defaults to 1.")
        ("trigger-mode,m", po::value<int>(),
         "Shutter trigger mode. Defaults to -1.\n\n"
         "Values:\n"
         " -1:  \tNo external trigger. Frames are captured in free-running mode at "
         "the currently set frame rate.\n"
         "  0:  \tStandard external trigger. Trigger edge causes sensor "
         "exposure, then sensor readout to internal memory.\n"
         "  1:  \tBulb shutter mode. Same as 0, except that sensor exposure "
         "duration is determined by trigger active duration.\n"
         " 13:  \tLow smear mode. Same as 0, speed of the vertical clock is "
         "increased near the end of the integration cycle.\n"
         " 14:  \tOverlapped exposure/readout external trigger. Sensor exposure "
         "occurs during sensory readout to internal memory. This is the "
         "fastest option.")
        ("trigger-rising,p",
         "True to trigger on rising edge, false to trigger on falling edge. "
         "Defaults to true.")
        ("trigger-pin,t", po::value<size_t>(),
         "GPIO pin number on that trigger is sent to if external shutter "
         "triggering is used. Defaults to 0.")
        ("roi,R", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "frame size. Defaults to full sensor size.")
        ("bin,b", po::value<std::string>(),
         "Two element array of unsigned ints, [bx,by], "
         "defining how pixels should be binned before transmission to the "
         "computer. Defaults to [1,1] (no binning).")
        ("white-balance,w", po::value<std::string>(),
         "Two element array of unsigned integers, [red,blue], used to "
         "specify the white balance. Values are between 0 and 1000. "
         "Only works for color sensors. Defaults to off.")
        ("auto-white-balance,W",
         "If specified, the white balance will be adjusted by the camera. "
         "This option overrides manual white-balance specification.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o : local_opts.options())
        config_keys_.push_back(o->long_name());
}

template <typename T>
void PointGreyCam<T>::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Camera index
    auto num_cams = findNumCameras();
    int index = 0;
    oat::config::getNumericValue<int>(
        vm, config_table, "index", index, 0, num_cams - 1);

    connectToCamera(index);
    turnCameraOn();

    // Frame rate
    if (oat::config::getNumericValue<double>(
            vm, config_table, "fps", frames_per_second_, 0.0))
        setupFrameRate(frames_per_second_, false);
    else
        setupFrameRate(frames_per_second_, true);

    // Shutter time
    double shutter_ms = 0.0;
    if (oat::config::getNumericValue<double>(
            vm, config_table, "shutter", shutter_ms, 0.0, 1000.0))
        setupShutter(shutter_ms, false);
    else
        setupShutter(shutter_ms, true);

    // Pixel color
    std::string col;
    if (oat::config::getValue<std::string>(vm, config_table, "color", col))
        pix_col_ = oat::str_color(col);

    // Determine if color conversion is required
    if (std::get<PG_FROM>(pix_map_.at(pix_col_)) != std::get<PG_TO>(pix_map_.at(pix_col_)))
        color_conversion_required_ = true;

    // Sensor gain
    double gain = 0.0;
    if (oat::config::getNumericValue<double>(
            vm, config_table, "gain", gain, 0.0))
        setupGain(gain, false);
    else
        setupGain(gain, true);

    // Set white balance
    // TODO: Needs three states: On, Auto, manual
    std::vector<double> wb;
    bool auto_wb;
    if (oat::config::getValue<bool>(vm, config_table, "auto-white-balance", auto_wb))
        setupWhiteBalance(0, 0, true, true);
    else if (oat::config::getArray<double, 2>(vm, config_table, "white-balance", wb))
        setupWhiteBalance(wb[0], wb[1], true, false);
    else
        setupWhiteBalance(0, 0, false, false);

    // Pixel binning
    std::vector<size_t> bin_size;
    if (oat::config::getArray<size_t, 2>(vm, config_table, "bin-size", bin_size))
        setupPixelBinning(bin_size[0], bin_size[1]);

    // ROI
    std::vector<size_t> roi;
    if (oat::config::getArray<size_t, 4>(vm, config_table, "roi", roi)) {

        use_roi_ = true;
        region_of_interest_.x      = roi[0];
        region_of_interest_.y      = roi[1];
        region_of_interest_.width  = roi[2];
        region_of_interest_.height = roi[3];

        setupImageFormat(roi);
    } else {
        setupImageFormat();
    }

    // TODO: Onboard frame buffer
    //int retries = 0;
    //if (oat::config::getValue(vm, config_table, "retries", retries, 0, false))
    //    setupCameraFrameBuffer(retries)

    // Enforce FPS
    oat::config::getValue<bool>(vm, config_table, "enforce_fps", enforce_fps_, false);

    // Strobe pin (configure before trigger to look for pin conflict)
    int strobe_pin = 1;
    oat::config::getNumericValue<int>(
        vm, config_table, "strobe-pin", strobe_pin, 0);
    setupStrobeOutput(strobe_pin);

    // Trigger mode
    int trigger_mode = -1;
    oat::config::getNumericValue<int>(
        vm, config_table, "trigger-mode", trigger_mode, -1);

    // Trigger polarity
    bool trigger_rising= true;
    oat::config::getNumericValue<bool>(
        vm, config_table, "trigger-rising", trigger_rising);

    // Trigger pin
    int trigger_pin = 0;
    oat::config::getNumericValue<int>(
        vm, config_table, "trigger-pin", trigger_pin, 0);

    if (trigger_pin == strobe_pin)
        throw rte("Stobe pin must be different from trigger pin.");

    setupAsyncTrigger(trigger_mode, trigger_rising, trigger_pin);

    // Start the configured camera
    setupGrabSettings();
    startCapture();

    // TODO: Has hack that requires camera to be running in order to function
    // Embed timestamp with frames
    setupEmbeddedImageData();
}

template <typename T>
void PointGreyCam<T>::connectToCamera(int index)
{
    auto num_cameras = static_cast<int>(findNumCameras());

    if (index >= num_cameras)
        throw (rte("Requested camera index " +
               std::to_string(index) + " is out of range.\n"));

    std::cout << "Connecting to camera: " << index << "\n";

    pg::BusManager busMgr;
    pg::PGRGuid guid;
    pg::Error error = busMgr.GetCameraFromIndex(index, &guid);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));
    error = camera_.Connect(&guid);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    printCameraInfo();

    std::cout << "Restoring default acqusition settings...\n";

    error = camera_.RestoreFromMemoryChannel(0);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Default settings restored.\n";
}

template <typename T>
void PointGreyCam<T>::setupFrameRate(double fps, bool is_auto)
{
    std::cout << "Setting up frame rate...\n";

    pg::Property prop;
    prop.type = pg::FRAME_RATE;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    prop.autoManualMode = is_auto;

    if (!is_auto) {
        prop.absValue = fps;
        std::cout << "Frame rate set to " + std::to_string(prop.absValue) + " FPS.\n";
    } else {
        std::cout << "Frame rate set to auto.\n";
    }

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    // If set to auto, then get the automatically configured frame frame rate
    if (is_auto) {
        error = camera_.GetProperty(&prop);
        if (error != pg::PGRERROR_OK)
            throw (rte(error.GetDescription()));

        frames_per_second_ = prop.absValue;
    }
}

template <typename T>
void PointGreyCam<T>::setupShutter(float shutter_ms, bool is_auto)
{
    std::cout << "Setting up shutter...\n";

    pg::Property prop;
    prop.type = pg::SHUTTER;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = shutter_ms;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    if (is_auto) {
        std::cout << "Shutter set to auto.\n";
    } else {
        std::cout << "Shutter time set to "
                  << std::fixed
                  << std::setprecision(2)
                  << shutter_ms << " ms.\n";
    }
}

template <typename T>
void PointGreyCam<T>::setupGain(float gain_db, bool is_auto)
{
    std::cout << "Setting camera gain...\n";

    pg::Property prop;
    prop.type = pg::GAIN;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = gain_db;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    if (is_auto) {
        std::cout << "Gain set to auto.\n";
    } else {
        std::cout << "Gain set to "
                  << std::fixed
                  << std::setprecision(2)
                  << gain_db << " dB.\n";
    }
}

template <typename T>
void PointGreyCam<T>::setupWhiteBalance(int bal_red, int bal_blue, bool is_on, bool is_auto)
{
    // Mono pixels do not support white balance
    if (is_on && pix_col_ == PIX_GREY) {
        std::cerr << oat::Warn(
            "You cannot adjust the white balance for mono frames.");
        return;
    }

    std::cout << "Setting camera white balance...\n";

    pg::Property prop;
    prop.type = pg::WHITE_BALANCE;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    prop.onOff = is_on;
    prop.autoManualMode = is_auto;
    prop.absControl = false;
    prop.valueA = bal_red;
    prop.valueB = bal_blue;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    if (is_on) {
        std::cout << "White balance set to: \n";
        std::cout << "\tRed: "
                  << std::fixed
                  << std::setprecision(2)
                  << bal_red << "\n";
        std::cout << "\tBlue: "
                  << std::fixed
                  << std::setprecision(2)
                  << bal_blue << "\n";
    } else {
        std::cout << "White balance turned off.\n";
    }
}

template <typename T>
void PointGreyCam<T>::setupImageFormat()
{
    static_assert(sizeof(T) == 0, "Not a valid PoinGreyCam type.");
}

template <typename T>
void PointGreyCam<T>::setupImageFormat(const std::vector<size_t> &roi_vec)
{
    (void)roi_vec; // Suppress unused parameter warnings
    static_assert(sizeof(T) == 0, "Not a valid PoinGreyCam type.");
}

template <typename T>
void PointGreyCam<T>::setupPixelBinning(size_t x_bin, size_t y_bin)
{
    (void)x_bin; // Suppress unused parameter warnings
    (void)y_bin;
    static_assert(sizeof(T) == 0, "Not a valid PoinGreyCam type.");
}

// template<typename T>
// int PointGreyCam<T>::setupCameraFrameBuffer() {
//
//    // If requested
//    if (use_frame_buffer_) {
//
//        unsigned int image_retransmit_reg;
//        const unsigned int image_retransmit_addr = 0x634;
//
//        pg::Error error = camera_.ReadRegister(image_retransmit_addr,
//                                              &image_retransmit_reg);
//        if (error != pg::PGRERROR_OK) {
//            throw (rte(error.GetDescription()));
//        }
//
//        std::cout << "Setting up camera_ frame buffering...\n";
//        // Enable framebuffer
//        image_retransmit_reg |= 1 << 0;
//
//        // Direct image data through frame buffer
//        image_retransmit_reg |= 1 << 1;
//
//        error = camera_.WriteRegister(image_retransmit_addr,
//        image_retransmit_reg);
//        if (error != pg::PGRERROR_OK) {
//            std::cout << "Warning: camera_ frame buffering requested, but this
//            "
//                         "camera_ does not support frame buffering.\n"
//                      << "Request ignored.\n";
//            use_frame_buffer_ = false;
//        }
//    }
//
//    return 0;
//}

// TODO: Required??
template <typename T>
void PointGreyCam<T>::turnCameraOn()
{
    // Power on the camera_
    const unsigned int k_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    pg::Error error = camera_.WriteRegister(k_cameraPower, k_powerVal);
    if (error != pg::PGRERROR_OK && error != pg::PGRERROR_NOT_IMPLEMENTED)
        throw (rte(error.GetDescription()));

    unsigned int regVal = 0;
    unsigned int retries = 10;

    // Wait for camera_ to complete power-up
    do {

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        error = camera_.ReadRegister(k_cameraPower, &regVal);
        if (error == pg::PGRERROR_TIMEOUT) {
            // ignore timeout errors, camera_ may not be responding to
            // register reads during power-up
        } else if (error != pg::PGRERROR_OK) {
            throw (rte(error.GetDescription()));
        }

        retries--;
    } while ((regVal & k_powerVal) == 0 && retries > 0);

    // Check for timeout errors after retrying
    if (error == pg::PGRERROR_TIMEOUT)
        throw (rte(error.GetDescription()));
}

template <typename T>
void PointGreyCam<T>::setupAsyncTrigger(int trigger_mode,
                                        bool trigger_rising,
                                        int trigger_pin)
{
    std::cout << "Setting up async trigger mode "<< trigger_mode << " \n";
    // Free Running
    if (trigger_mode < 0 ) {
        use_trigger_ = false;
        return;
    }

    if (trigger_mode != 0  ||
        trigger_mode != 1  ||
        trigger_mode != 13 ||
        trigger_mode != 14) {
      //throw (rte("Trigger mode is unsupported."));
    }

    // Get current trigger settings
    pg::TriggerModeInfo trigger_mode_info;
    pg::Error error = camera_.GetTriggerModeInfo(&trigger_mode_info);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    if (trigger_mode_info.present != true)
        throw (rte(error.GetDescription()));

    pg::TriggerMode triggerMode;
    error = camera_.GetTriggerMode(&triggerMode);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    triggerMode.onOff = true;
    triggerMode.polarity = trigger_rising;
    triggerMode.mode = trigger_mode;
    triggerMode.parameter = 0;
    triggerMode.source = trigger_pin;

    error = camera_.SetTriggerMode(&triggerMode);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    // Trigger info
    std::cout << "Trigger the camera by sending a trigger pulse to GPIO_"
              << triggerMode.source << "\n";

    use_trigger_ = true;
}

template <typename T>
void PointGreyCam<T>::setupGrabSettings()
{
    // Setup frame buffering
    // NOTE: For some reason grabMode = pg::BUFFER_FRAMES does not play nicely
    // with the time-stamp correction for dropped triggers. I have yet to
    // understand why.
    std::cout << "setting up grab...\n";
    pg::FC2Config flyCapConfig;
    pg::Error error = camera_.GetConfiguration(&flyCapConfig);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    flyCapConfig.grabTimeout = 10;
    flyCapConfig.grabMode = pg::DROP_FRAMES;
    flyCapConfig.highPerformanceRetrieveBuffer = true;
    //flyCapConfig.numBuffers = 1;

    error = camera_.SetConfiguration(&flyCapConfig);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    // TODO: Custom frame rate
    // VideoMode videoMode;
    // FrameRate frameRate;
    // camera_.GetVideoModeAndFrameRate ( &videoMode, &frameRate);

    // TODO: This hangs...
    // Poll to ensure camera_ is ready
    //    if (use_trigger_) { // If false, camera_ will free run
    //        bool retVal = pollForTriggerReady();
    //        if (!retVal) {
    //            std::cout << "\n";
    //            std::cout << "Error polling for trigger ready. Exiting...\n";
    //            exit(EXIT_FAILURE);
    //        }
    //    }
}

template <typename T>
void PointGreyCam<T>::startCapture()
{
    std::cout << "Starting capture...\n";
    // Camera is ready, start capturing images
    pg::Error error = camera_.StartCapture();
    if (error == pg::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) { 
        throw (rte("Interface bandwidth exceeded. Cannot start camera_..\n"));
    } else if (error != pg::PGRERROR_OK) {
        std::cout << "Error starting capture: \n";
        throw (rte(error.GetDescription()));
    }

    acquisition_started_ = true;
}

template <typename T>
void PointGreyCam<T>::setupStrobeOutput(int strobe_pin)
{
    pg::StrobeControl strobe;
    strobe.source = strobe_pin;
    strobe.onOff = true;
    strobe.polarity = 1;
    strobe.delay = 0.0f;
    strobe.duration = 0.0f;
    //camera_.WriteRegister(0x19D0, 0x80000001); // start strobe on Blackfly: needed for strobe output in my model --mmyros

    pg::Error error = camera_.SetStrobe(&strobe);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));
}

template <typename T>
void PointGreyCam<T>::setupEmbeddedImageData()
{
    pg::EmbeddedImageInfo embeddedInfo;
    pg::Error error = camera_.GetEmbeddedImageInfo(&embeddedInfo);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    // For now, only inlcude timestamp
    embeddedInfo.timestamp.onOff = true;

    error = camera_.SetEmbeddedImageInfo(&embeddedInfo);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    // TODO: HACK! See https://github.com/jonnew/Oat/issues/11
    int i = 0;
    while (use_trigger_ && (error == pg::PGRERROR_OK || i < 10)) {
        i++;
    }
}

// TODO: event driven acquisition.
//void PointGreyCam<T>::onGrabbedImage(Image* pImage, const void* pCallbackData) {
//
//    &raw_image_ = pImage;
//    current_frame = imageToMat();
//    serveMat();
//
//}

// TODO: implement onboard buffer and perform retry a RetrieveBuffer
// A single time if a torn image is detected.
template <typename T>
int PointGreyCam<T>::grabImage(pg::Image *raw_image)
{
    assert (acquisition_started_ &&
            "Cannot grab image because acquisition has not been started.");

    pg::Error error;
    error = camera_.RetrieveBuffer(raw_image);

    if (error == pg::PGRERROR_TIMEOUT) {
        return -1;
#ifndef NDEBUG
    } else if (error == pg::PGRERROR_IMAGE_CONSISTENCY_ERROR) {
        std::cerr << oat::Error("WARNING: torn image detected.\n");
#endif
    } else if (error != pg::PGRERROR_OK) {
        printError(error);
        std::cerr << oat::Error("WARNING: Point Grey capture errored with type "
                                + std::to_string(error.GetType()) + "\n");
    }

    // Get the embedded timestamp of the must current frame
    const pg::TimeStamp ts = raw_image->GetTimeStamp();

    if (!ieee_1394_start_set_) {
        ieee_1394_start_cycle_ = ts.cycleSeconds * IEEE_1394_HZ + ts.cycleCount ;
        ieee_1394_start_set_ = true;
    }

    uint64_t total_ieee_1394_cycles =
        uncycle1394Timestamp(ts.cycleSeconds, ts.cycleCount);

    // Convert to chrono::time_point
    tock_ = tick_;
    tick_ = std::chrono::duration_cast<oat::Sample::Microseconds> (
                oat::Sample::IEEE1394Tick(total_ieee_1394_cycles)
            );

    // Calculate the delay since the last frame was acquired.
    auto delay = static_cast<double>(((tick_ - tock_).count()) / 1.0e6);

    if (first_frame_ || !enforce_fps_) {
        first_frame_ = false;
        return 0;
    } else if (delay > 0.0) {
        // Return the number of skipped frames. This should be 0, but PG cameras
        // reject triggers on some ocations and we need to fill in the blanks to
        // prevent offsets...
        return static_cast<int>(std::round(frames_per_second_ * delay  - 1.0));
    } else {
        return 0;
    }
}

template <typename T>
uint64_t PointGreyCam<T>::uncycle1394Timestamp(int ieee_1394_sec,
                                               int ieee_1394_cycle)
{
    if (ieee_1394_sec - last_ieee_1394_sec_ < 0)
        ieee_1394_cycle_index_++;

    last_ieee_1394_sec_ = ieee_1394_sec;

    return (uint64_t)(ieee_1394_cycle_index_ * 128 + ieee_1394_sec) * IEEE_1394_HZ
           + ieee_1394_cycle
           - ieee_1394_start_cycle_;
}

template <typename T>
void PointGreyCam<T>::connectToNode()
{
    static_assert(sizeof(T) == 0, "Not a valid PoinGreyCam type.");
}

template <typename T>
bool PointGreyCam<T>::process()
{
    pg::Image raw_image;
    int rc = grabImage(&raw_image);

    // There was a grab timeout.
    // Allow check to see if SIGINT occurred.
    if (rc == -1)
        return false;

    if (rc > 0) {
        std::cerr << oat::Warn("Frame re-transmission due to " +
                               std::to_string(rc) +
                               " skipped trigger(s).\n");
    }

    int i = 0;
    do {

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        frame_sink_.wait();

        if (color_conversion_required_)
            raw_image.Convert(std::get<PG_TO>(pix_map_.at(pix_col_)), shmem_image_.get());
        else
            shmem_image_->DeepCopy(&raw_image);

        shared_frame_.incrementSampleCount(tick_);

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

    } while (i++ < rc);

    return false;
}

template <typename T>
unsigned int PointGreyCam<T>::findNumCameras(void)
{
    pg::Error error;
    pg::BusManager busMgr;

    unsigned int num_cameras = 0;

    error = busMgr.GetNumOfCameras(&num_cameras);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    return num_cameras;
}

template <typename T>
void PointGreyCam<T>::printError(pg::Error error)
{
    error.PrintErrorTrace();
    if (!camera_.IsConnected())
        std::cerr << "Camera must be connected before getting its info.\n";
}

template <typename T>
bool PointGreyCam<T>::pollForTriggerReady()
{
    const unsigned int k_softwareTrigger = 0x62C;
    pg::Error error;
    unsigned int regVal = 0;

    do {
        error = camera_.ReadRegister(k_softwareTrigger, &regVal);
        if (error != pg::PGRERROR_OK)
            throw (rte(error.GetDescription()));

    } while ((regVal >> 31) != 0);

    return true;
}

template <typename T>
void PointGreyCam<T>::printCameraInfo(void)
{
    pg::CameraInfo camera_info;
    pg::Error error = camera_.GetCameraInfo(&camera_info);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::ostringstream macAddress;
    macAddress
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[0] << ":"
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[1] << ":"
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[2] << ":"
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[3] << ":"
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[4] << ":"
        << std::hex << std::setw(2) << std::setfill('0')
        << (unsigned int) camera_info.macAddress.octets[5];


    std::ostringstream ipAddress;
    ipAddress
        << (unsigned int) camera_info.ipAddress.octets[0] << "."
        << (unsigned int) camera_info.ipAddress.octets[1] << "."
        << (unsigned int) camera_info.ipAddress.octets[2] << "."
        << (unsigned int) camera_info.ipAddress.octets[3];

    std::ostringstream subnetMask;
    subnetMask
        << (unsigned int) camera_info.subnetMask.octets[0] << "."
        << (unsigned int) camera_info.subnetMask.octets[1] << "."
        << (unsigned int) camera_info.subnetMask.octets[2] << "."
        << (unsigned int) camera_info.subnetMask.octets[3];

    std::ostringstream defaultGateway;
    defaultGateway
        << (unsigned int) camera_info.defaultGateway.octets[0] << "."
        << (unsigned int) camera_info.defaultGateway.octets[1] << "."
        << (unsigned int) camera_info.defaultGateway.octets[2] << "."
        << (unsigned int) camera_info.defaultGateway.octets[3];

    std::cout << "\n";
    std::cout << "*** GENERAL CAMERA INFORMATION ***\n";
    std::cout << "Serial number: " << camera_info.serialNumber << "\n";
    std::cout << "Camera model: " << camera_info.modelName << "\n";
    std::cout << "Camera vendor: " << camera_info.vendorName << "\n";
    std::cout << "Sensor: " << camera_info.sensorInfo << "\n";
    std::cout << "Resolution: " << camera_info.sensorResolution << "\n";
    std::cout << "Firmware version: " << camera_info.firmwareVersion << "\n";
    std::cout << "Firmware build time: " << camera_info.firmwareBuildTime << "\n\n";

    std::cout << "*** CAMERA INTERFACE INFORMATION ***\n";
    std::cout << "GigE version :"
              << camera_info.gigEMajorVersion << "."
              << camera_info.gigEMinorVersion << "\n";
    std::cout << "User defined name :" << camera_info.userDefinedName << "\n";
    //std::cout << "XML URL 1: " << camera_info.xmlURL1 << "\n";
    //std::cout << "XML URL 2: " << camera_info.xmlURL2 << "\n";
    std::cout << "MAC address: " << macAddress.str() << "\n";
    std::cout << "IP address: " << ipAddress.str() << "\n";
    std::cout << "Subnet mask: " << subnetMask.str() << "\n";
    std::cout << "Default gateway: " << defaultGateway.str() << "\n\n";
}

/* SPECIALIZATIONS */

// 0. GigE Camera

template <>
void PointGreyCam<pg::GigECamera>::connectToCamera(int index)
{
    auto num_cameras = static_cast<int>(findNumCameras());

    if (index >= num_cameras)
        throw (rte("Requested camera index " +
               std::to_string(index) + " is out of range.\n"));

    std::cout << "Connecting to camera: " << index << "\n";

    pg::BusManager busMgr;
    pg::PGRGuid guid;
    pg::Error error = busMgr.GetCameraFromIndex(index, &guid);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    error = camera_.Connect(&guid);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    printCameraInfo();

    std::cout << "Restoring default acqusition settings...\n";

    error = camera_.RestoreFromMemoryChannel(0);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Default settings restored.\n";

    unsigned int numStreamChannels = 0;
    error = camera_.GetNumStreamChannels(&numStreamChannels);
    std::cout << "Default settings restored.\n";
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    for (unsigned int i = 0; i < numStreamChannels; i++) {
        pg::GigEStreamChannel streamChannel;
        error = camera_.GetGigEStreamChannelInfo(i, &streamChannel);
        if (error != pg::PGRERROR_OK)
            throw (rte(error.GetDescription()));

        // TODO: This is not going to be valid if cameras are on a switch...
        streamChannel.destinationIpAddress.octets[0] = 224;
        streamChannel.destinationIpAddress.octets[1] = 0;
        streamChannel.destinationIpAddress.octets[2] = 0;
        streamChannel.destinationIpAddress.octets[3] = 1;

        // TODO: Make a more reasoned choice for these parameters based on the
        // number of cameras on the system...
        streamChannel.packetSize = 9000;
        streamChannel.interPacketDelay = 250;

        error = camera_.SetGigEStreamChannelInfo(i, &streamChannel);
        if (error != pg::PGRERROR_OK)
            throw (rte(error.GetDescription()));

        std::cout << "Stream channel information for channel " << i << "\n";

        std::ostringstream ipAddress;
        ipAddress
            << (unsigned int) streamChannel.destinationIpAddress.octets[0] << "."
            << (unsigned int) streamChannel.destinationIpAddress.octets[1] << "."
            << (unsigned int) streamChannel.destinationIpAddress.octets[2] << "."
            << (unsigned int) streamChannel.destinationIpAddress.octets[3];

        std::cout << "Network interface: " << streamChannel.networkInterfaceIndex
                  << "\n";
        //std::cout << "Host Port: " << streamChannel.hostPort << "\n";
        std::cout << "Do not fragment bit: "
                  << (streamChannel.doNotFragment ? "Enabled" : "Disabled")
                  << "\n";
        std::cout << "Packet size: " << streamChannel.packetSize << "\n";
        std::cout << "Inter packet delay: " << streamChannel.interPacketDelay
                  << "\n";
        std::cout << "Destination IP address: " << ipAddress.str() << "\n";
        std::cout << "Source port (on camera): " << streamChannel.sourcePort
                  << "\n\n";
    }
}

template <>
void PointGreyCam<pg::GigECamera>::setupImageFormat()
{
    std::cout << "Setting image parameters...\n";

    pg::GigEImageSettingsInfo image_settings_info;
    pg::Error error = camera_.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    pg::GigEImageSettings imageSettings;
    imageSettings.offsetX = 0;
    imageSettings.offsetY = 0;
    imageSettings.width   = image_settings_info.maxWidth;
    imageSettings.height  = image_settings_info.maxHeight;
    imageSettings.pixelFormat = std::get<PG_FROM>(pix_map_.at(pix_col_));

    std::cout << "ROI set to [0 0 "
              << image_settings_info.maxWidth
              << " "
              << image_settings_info.maxHeight
              << "]\n";

    error = camera_.SetGigEImageSettings(&imageSettings);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Image settings configured.\n";
}

template <>
void PointGreyCam<pg::GigECamera>::setupImageFormat(const std::vector<size_t> &roi_vec)
{
    std::cout << "Querying image settings information...\n";

    pg::GigEImageSettingsInfo image_settings_info;
    pg::Error error = camera_.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    if (roi_vec[0] > image_settings_info.maxWidth ||
        roi_vec[1] > image_settings_info.maxHeight) {
        throw (rte("ROI pixel offsets are larger than the sensor array. Exiting.\n"));
    }

    if ((roi_vec[2] + roi_vec[0]) > image_settings_info.maxWidth)
        throw (rte("X-axis ROI settings are off the sensor array\n"));

    if ((roi_vec[3] + roi_vec[1]) > image_settings_info.maxHeight)
        throw (rte("Y-axis ROI settings are off the sensor array\n"));

    pg::GigEImageSettings imageSettings;
    imageSettings.offsetX = roi_vec[0];
    imageSettings.offsetY = roi_vec[1];
    imageSettings.height  = roi_vec[2];
    imageSettings.width   = roi_vec[3];
    imageSettings.pixelFormat = std::get<PG_FROM>(pix_map_.at(pix_col_)); //pg::PIXEL_FORMAT_RAW8;

    std::cout << "ROI set to ["
              << roi_vec[0]
              << " "
              << roi_vec[1]
              << " "
              << roi_vec[2]
              << " "
              << roi_vec[3]
              << "]\n";

    error = camera_.SetGigEImageSettings(&imageSettings);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));
}

template <>
void PointGreyCam<pg::GigECamera>::setupPixelBinning(size_t x_bin, size_t y_bin)
{
    std::cout << "Setting image binning...\n";

    // On-board image binning
    pg::Error error;
    error = camera_.SetGigEImageBinningSettings(x_bin, y_bin);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Onboard binning set to [" << x_bin << " " << y_bin << "]\n";
}

template <>
void PointGreyCam<pg::GigECamera>::connectToNode()
{
    pg::GigEImageSettings imageSettings;

    pg::Error error = camera_.GetGigEImageSettings(&imageSettings);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    pg::Image temp(imageSettings.height,
                   imageSettings.width,
                   std::get<PG_TO>(pix_map_.at(pix_col_)));

    const size_t bytes = temp.GetDataSize();
    const size_t rows = temp.GetRows();
    const size_t cols = temp.GetCols();
    const size_t stride = temp.GetStride();

    frame_sink_.bind(frame_sink_address_, bytes);

    shared_frame_ = frame_sink_.retrieve(rows, cols, std::get<CV_TYPE>(pix_map_.at(pix_col_)), pix_col_);
    shared_frame_.set_rate_hz(frames_per_second_);

    // Use the shared_frame_.data, which points to a block of shared memory as
    // rbg_image's data buffer. When changes are made to shmem_image_, this is
    // automatically propagated into shmem and 'converted' into a cv::Mat
    // (although this 'conversion' is simply filling in appropriate header info,
    // which was accomplished in the call to frame_sink_.retrieve())
    shmem_image_ = oat::make_unique<pg::Image>
            (rows, cols, stride, shared_frame_.data, bytes, std::get<PG_TO>(pix_map_.at(pix_col_)));
}

// 1. USB Camera

template <>
void PointGreyCam<pg::Camera>::setupImageFormat(
    const std::vector<size_t> &roi_vec)
{
    std::cout << "Setting image parameters...\n";

    const pg::Mode k_fmt7Mode = pg::MODE_0;
    pg::Format7Info image_settings_info;
    bool supported;

    image_settings_info.mode = k_fmt7Mode;
    pg::Error error = camera_.GetFormat7Info(&image_settings_info, &supported);
    if (error != pg::PGRERROR_OK)
         throw (rte(error.GetDescription()));

    const pg::PixelFormat k_fmt7PixFmt = std::get<PG_FROM>(pix_map_.at(pix_col_));

    if ((k_fmt7PixFmt & image_settings_info.pixelFormatBitField) == 0) {
        std::cerr << "Pixel format is not supported\n";
        throw(rte(error.GetDescription()));
    }

    if (roi_vec[0] > image_settings_info.maxWidth ||
        roi_vec[1] > image_settings_info.maxHeight) {
        throw (rte("ROI pixel offsets are larger than the sensor array. Exiting.\n"));
    }

    if ((roi_vec[2] + roi_vec[0]) > image_settings_info.maxWidth)
        throw (rte("X-axis ROI settings are off the sensor array\n"));

    if ((roi_vec[3] + roi_vec[1]) > image_settings_info.maxHeight)
        throw (rte("Y-axis ROI settings are off the sensor array\n"));

    pg::Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = image_settings_info.mode;
    fmt7ImageSettings.offsetX = roi_vec[0];
    fmt7ImageSettings.offsetY = roi_vec[1];
    fmt7ImageSettings.width   = roi_vec[2];
    fmt7ImageSettings.height  = roi_vec[3];
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;

    std::cout << "ROI set to ["
              << roi_vec[0]
              << " "
              << roi_vec[1]
              << " "
              << roi_vec[2]
              << " "
              << roi_vec[3]
              << "]\n";

    pg::Format7PacketInfo fmt7PacketInfo;
    bool valid = true;

    // Validate the settings to make sure that they are valid
    error = camera_.ValidateFormat7Settings(
        &fmt7ImageSettings, &valid, &fmt7PacketInfo);

    if (error != pg::PGRERROR_OK)
        throw(rte(error.GetDescription()));

    if (!valid)
        throw (rte("Format7 image settings are invalid for this camera."));

    error = camera_.SetFormat7Configuration(
        &fmt7ImageSettings,
        fmt7PacketInfo.recommendedBytesPerPacket );

    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Image settings configured.\n";
}

template <>
void PointGreyCam<pg::Camera>::setupImageFormat()
{
    std::cout << "Setting image parameters...\n";

    const pg::Mode k_fmt7Mode = pg::MODE_0;

    pg::Format7Info image_settings_info;
    bool supported;
    image_settings_info.mode = k_fmt7Mode;
    pg::Error error = camera_.GetFormat7Info(&image_settings_info, &supported);
    if (error != pg::PGRERROR_OK)
         throw (rte(error.GetDescription()));

    const pg::PixelFormat k_fmt7PixFmt = std::get<PG_FROM>(pix_map_.at(pix_col_));

    if ((k_fmt7PixFmt & image_settings_info.pixelFormatBitField) == 0) {
        std::cerr << "Pixel format is not supported\n";
        throw(rte(error.GetDescription()));
    }

    pg::Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = image_settings_info.mode;
    fmt7ImageSettings.offsetX = 0;
    fmt7ImageSettings.offsetY = 0;
    fmt7ImageSettings.width = image_settings_info.maxWidth;
    fmt7ImageSettings.height = image_settings_info.maxHeight;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;

    std::cout << "ROI set to [0 0 "
              << image_settings_info.maxWidth
              << " "
              << image_settings_info.maxHeight
              << "]\n";

    pg::Format7PacketInfo fmt7PacketInfo;
    bool valid = true;

    // Validate the settings to make sure that they are valid
    error = camera_.ValidateFormat7Settings(
        &fmt7ImageSettings, &valid, &fmt7PacketInfo);

    if (error != pg::PGRERROR_OK)
        throw(rte(error.GetDescription()));

    if (!valid)
        throw (rte("Format7 image settings are invalid for this camera."));

    error = camera_.SetFormat7Configuration(
        &fmt7ImageSettings,
        fmt7PacketInfo.recommendedBytesPerPacket );

    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Image settings configured.\n";
}

template <>
void PointGreyCam<pg::Camera>::setupPixelBinning(size_t x_bin, size_t y_bin)
{
    (void)x_bin; // Prevent unused parameter warning
    (void)y_bin;
    std::cout << "Pixel binning is not implemented.\n";
}

template <>
void PointGreyCam<pg::Camera>::connectToNode()
{
    pg::Format7ImageSettings pImageSettings;
    unsigned int pPacketSize;
    float pPercentage;

    pg::Error error = camera_.GetFormat7Configuration(
        &pImageSettings, &pPacketSize, &pPercentage);

    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    pg::Image temp(pImageSettings.height,
                   pImageSettings.width,
                   std::get<PG_TO>(pix_map_.at(pix_col_)));

    const size_t bytes = temp.GetDataSize();
    const size_t rows = temp.GetRows();
    const size_t cols = temp.GetCols();
    const size_t stride = temp.GetStride();

    frame_sink_.bind(frame_sink_address_, bytes);

    shared_frame_ = frame_sink_.retrieve(rows, cols, std::get<CV_TYPE>(pix_map_.at(pix_col_)), pix_col_);
    shared_frame_.set_rate_hz(frames_per_second_);

    // Use the shared_frame_.data, which points to a block of shared memory as
    // rbg_image's data buffer. When changes are made to shmem_image_, this is
    // automatically propagated into shmem and 'converted' into a cv::Mat
    // (although this 'conversion' is simply filling in appropriate header info,
    // which was accomplished in the call to frame_sink_.retrieve())
    shmem_image_ = oat::make_unique<pg::Image>
            (rows, cols, stride, shared_frame_.data, bytes, std::get<PG_TO>(pix_map_.at(pix_col_)));
}

// Explicit instantiation
template class PointGreyCam<pg::Camera>;
template class PointGreyCam<pg::GigECamera>;

} /* namespace oat */
