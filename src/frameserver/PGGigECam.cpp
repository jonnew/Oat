//******************************************************************************
//* File:   PGGigECam.cpp
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

#include "PGGigECam.h"

#include <chrono>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include <cpptoml.h>
#include "FlyCapture2.h"
#include <opencv2/opencv.hpp>

#include "../../lib/utility/make_unique.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

PGGigECam::PGGigECam(const std::string &sink_address) :
  FrameServer(sink_address)
{
    // Available options
    config_keys_ = {"index",
                    "fps",
                    "shutter",
                    "gain",
                    "white_bal",
                    "roi",
                    "bin",
                    "trigger_mode",
                    "trigger_rising",
                    "trigger_pin",
                    "enforce_fps",
                    "strobe_pin"};

    // Intialize frame timing
    tick_ = oat::Sample::Microseconds(0);
    tock_ = oat::Sample::Microseconds(0);
}

PGGigECam::~PGGigECam() {

    // Ignore error return values -- throwing exception unsafe in destructor
    camera_.StopCapture();
    camera_.Disconnect();
}

///**
// * Set default camera_ configuration
// */
//void PGGigECam::configure() {
//
//    setCameraIndex(index_);
//    connectToCamera();
//    turnCameraOn();
//    setupStreamChannels();
//    setupFrameRate(frames_per_second_, true);
//    //setupExposure(true);
//    setupShutter(true);
//    setupGain(true);
//    setupWhiteBalance(false);
//    setupDefaultImageFormat();
//    setupTrigger();
//    //setupCameraFrameBuffer();
//    setupEmbeddedImageData();
//}
//
void PGGigECam::appendOptions(po::options_description &opts) const {

    // Accepts default options
    FrameServer::appendOptions(opts);

    // Update CLI options
    opts.add_options()
        ("index,i", po::value<size_t>(),
         "")
        ("fps,r", po::value<double>(),
         "Frames to serve per second.")
        ("enforce-fps,e", po::value<bool>(),
         "")
        ("shutter,s", po::value<double>(),
         "")
        ("gain,g", po::value<double>(),
         "")
        ("strobe-pin,S", po::value<size_t>(),
         "")
        ("trigger-mode,m", po::value<int>(),
         "")
        ("trigger-rising,p", po::value<bool>(),
         "")
        ("trigger-pin,t", po::value<size_t>(),
         "")
        ("bin {CF}", po::value<std::string>(),
         "")
        ("white-balance {CF}", po::value<std::string>(),
         "")
        ;
}

void PGGigECam::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Camera index
    auto num_cams = findNumCameras();
    size_t index = 0;
    oat::config::getNumericValue<size_t>(
        vm, config_table, "index", index, 0, num_cams - 1
    );

    connectToCamera(index);
    turnCameraOn();
    setupStreamChannels();

    // Frame rate
    if (oat::config::getNumericValue(vm, config_table, "fps", frames_per_second_, 0.0))
        setupFrameRate(frames_per_second_, false);
    else
        setupFrameRate(frames_per_second_, true);

    // Shutter time
    double shutter_ms = 0.0;
    if (oat::config::getNumericValue(vm, config_table, "shutter", shutter_ms, 0.0, 1000.0))
        setupShutter(shutter_ms, false);
    else
        setupShutter(shutter_ms, true);

    // Sensor gain
    double gain = 0.0;
    if (oat::config::getNumericValue(vm, config_table, "gain", gain, 0.0))
        setupGain(gain, false);
    else
        setupGain(gain, true);

    // Set white balance
    oat::config::Array wb;
    if (oat::config::getArray(config_table, "white-bal", wb, 2, false)) {
        auto wb_vec = wb->array_of<double>();
        setupWhiteBalance(wb_vec[0]->get(), 
                          wb_vec[1]->get(), 
                          true);
    } else {
        setupWhiteBalance(0, 0, false);
    }

    // Pixel binning
    oat::config::Array bin_size;
    if (oat::config::getArray(config_table, "bin-size", bin_size, 2, false)) {
        auto bin_vec = bin_size->array_of<int64_t>();
        setupPixelBinning(static_cast<size_t>(bin_vec[0]->get()), 
                          static_cast<size_t>(bin_vec[1]->get()));
    }

    // ROI
    oat::config::Array roi;
    if (oat::config::getArray(config_table, "roi", roi, 4, false)) {
        auto roi_arr = roi->array_of<int64_t>();
        std::vector<size_t> roi_vec {static_cast<size_t>(roi_arr[0]->get()), 
                                     static_cast<size_t>(roi_arr[1]->get()), 
                                     static_cast<size_t>(roi_arr[2]->get()), 
                                     static_cast<size_t>(roi_arr[3]->get())};
        setupImageFormat(roi_vec);
    } else {
        setupImageFormat();
    }

    // TODO: Onboard frame buffer
    //int retries = 0;
    //if (oat::config::getValue(vm, config_table, "retries", retries, 0, false))
    //    setupCameraFrameBuffer(retries)

    // Enforce FPS
    oat::config::getValue(vm, config_table, "enforce_fps", enforce_fps_, false);

    // Strobe pin (configure before trigger to look for pin conflict)
    int strobe_pin = 1;
    oat::config::getNumericValue(vm, config_table, "strobe-pin", strobe_pin, 0);
    setupStrobeOutput(strobe_pin);

    // Trigger mode
    int trigger_mode = -1;
    oat::config::getNumericValue(vm, config_table, "trigger-mode", trigger_mode, -1);

    // Trigger polarity
    bool trigger_rising= true;
    oat::config::getNumericValue(vm, config_table, "trigger-rising", trigger_rising);

    // Trigger pin
    int trigger_pin = 0;
    oat::config::getNumericValue(vm, config_table, "trigger-pin", trigger_pin , 0);

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

void PGGigECam::connectToCamera(size_t index) {

    auto num_cameras = findNumCameras();

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

void PGGigECam::setupStreamChannels() {

    unsigned int numStreamChannels = 0;
    pg::Error error = camera_.GetNumStreamChannels(&numStreamChannels);
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

        std::cout << "Printing stream channel information for channel " << i << "\n";
        printStreamChannelInfo(&streamChannel);
    }
}

void PGGigECam::setupFrameRate(double fps, bool is_auto) {

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

void PGGigECam::setupShutter(float shutter_ms, bool is_auto) {

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

void PGGigECam::setupGain(float gain_db, bool is_auto) {

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

    if (is_auto)
        std::cout << "Gain set to auto.\n";
    else
        std::cout << "Gain set to " << std::fixed << std::setprecision(2) << gain_db << " dB.\n";
}

void PGGigECam::setupWhiteBalance(int bal_red,
                                  int bal_blue,
                                  bool is_on) {

    std::cout << "Setting camera white balance...\n";

    pg::Property prop;
    prop.type = pg::WHITE_BALANCE;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (rte(error.GetDescription()));
    }

    prop.onOff = is_on;
    prop.autoManualMode = false;
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

/**
 * Default image setup. Image uses all available pixels and BRG pixel format.
 *
 * @return 0 if successful.
 */
void PGGigECam::setupImageFormat() {

    std::cout << "Querying image setting information...\n";

    pg::GigEImageSettingsInfo image_settings_info;
    pg::Error error = camera_.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    pg::GigEImageSettings imageSettings;
    imageSettings.offsetX = 0;
    imageSettings.offsetY = 0;
    imageSettings.width   = image_settings_info.maxWidth;
    imageSettings.height  = image_settings_info.maxHeight;
    imageSettings.pixelFormat = pg::PIXEL_FORMAT_RAW8;
    std::cout << imageSettings.pixelFormat << std::endl;
    // TODO: Camera model dependent!!!

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

/**
 * Custom image setup. Image uses the internally specified ROI settings.
 * @return
 */
void PGGigECam::setupImageFormat(const std::vector<size_t> &roi_vec) {

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
        throw (rte("Current X-axis ROI settings are off the sensor array\n"));

    if ((roi_vec[3] + roi_vec[1]) > image_settings_info.maxHeight)
        throw (rte("Current Y-axis ROI settings are off the sensor array\n"));

    pg::GigEImageSettings imageSettings;
    imageSettings.offsetX = roi_vec[0];
    imageSettings.offsetY = roi_vec[1];
    imageSettings.height  = roi_vec[2];
    imageSettings.width   = roi_vec[3];
    imageSettings.pixelFormat = pg::PIXEL_FORMAT_RAW8;

    std::cout << "ROI set to [0 0 " 
              << image_settings_info.maxWidth
              << " " 
              << image_settings_info.maxHeight
              << "]\n";

    error = camera_.SetGigEImageSettings(&imageSettings);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));
}

int PGGigECam::setupPixelBinning(size_t x_bin, size_t y_bin) {

    std::cout << "Setting image binning...\n";

    // On-board image binning
    pg::Error error;
    error = camera_.SetGigEImageBinningSettings(x_bin, y_bin);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    std::cout << "Onboard binning set to [" << x_bin << " " << y_bin << "]\n";

    return 0;
}

//int PGGigECam::setupCameraFrameBuffer() {
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
//        error = camera_.WriteRegister(image_retransmit_addr, image_retransmit_reg);
//        if (error != pg::PGRERROR_OK) {
//            std::cout << "Warning: camera_ frame buffering requested, but this "
//                         "camera_ does not support frame buffering.\n"
//                      << "Request ignored.\n";
//            use_frame_buffer_ = false;
//        }
//    }
//
//    return 0;
//}

/**
 * Once connected to the camera_, issue power on command.
 *
 * @return 0 if successful.
 */
void PGGigECam::turnCameraOn() {

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

void PGGigECam::setupAsyncTrigger(int trigger_mode, 
                                  bool trigger_rising, 
                                  int trigger_pin) {

    // Free Running
    if (trigger_mode < 0 ) {
        use_trigger_ = false;
        return;
    }

    if (trigger_mode == -7)
        use_software_trigger_ = true;

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
    if (use_software_trigger_)
        std::cout << "Trigger the camera by pressing Enter\n";
    else
        std::cout << "Trigger the camera by sending a trigger pulse to GPIO_"
                  << triggerMode.source << "\n";
}

void PGGigECam::setupGrabSettings() {

    // Setup frame buffering
    // NOTE: For some reason grabMode = pg::BUFFER_FRAMES does not play nicely
    // with the time-stamp correction for dropped triggers. I have yet to
    // understand why.
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

void PGGigECam::startCapture() {

    // Camera is ready, start capturing images
    pg::Error error = camera_.StartCapture();
    if (error == pg::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED)
        throw (rte("Interface bandwidth exceeded. Cannot start camera_..\n"));
    else if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    acquisition_started_ = true;
}

void PGGigECam::setupStrobeOutput(int strobe_pin) {

    pg::StrobeControl strobe;
    strobe.source = strobe_pin;
    strobe.onOff = true;
    strobe.polarity = 1;
    strobe.delay = 0.0f;
    strobe.duration = 0.0f;

    pg::Error error = camera_.SetStrobe(&strobe);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

}

void PGGigECam::setupEmbeddedImageData() {

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
//void PGGigECam::onGrabbedImage(Image* pImage, const void* pCallbackData) {
//
//    &raw_image_ = pImage;
//    current_frame = imageToMat();
//    serveMat();
//
//}

// TODO: implement onboard buffer and perform retry a RetrieveBuffer
// A single time if a torn image is detected.
int PGGigECam::grabImage() {

    assert (acquisition_started_ &&
            "Cannot grab image because acquisition has not been started.");

    pg::Error error;
    error = camera_.RetrieveBuffer(&raw_image_);

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
    const pg::TimeStamp ts = raw_image_.GetTimeStamp();

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

uint64_t PGGigECam::uncycle1394Timestamp(int ieee_1394_sec,
                                         int ieee_1394_cycle) {

    if (ieee_1394_sec - last_ieee_1394_sec_ < 0)
        ieee_1394_cycle_index_++;

    last_ieee_1394_sec_ = ieee_1394_sec;

    return (uint64_t)(ieee_1394_cycle_index_ * 128 + ieee_1394_sec) * IEEE_1394_HZ
           + ieee_1394_cycle
           - ieee_1394_start_cycle_;
}

void PGGigECam::connectToNode() {

    pg::GigEImageSettings imageSettings;

    pg::Error error = camera_.GetGigEImageSettings(&imageSettings);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    pg::Image temp(imageSettings.height,
                   imageSettings.width,
                   pg::PIXEL_FORMAT_BGR);

    raw_image_.Convert(pg::PIXEL_FORMAT_BGR, &temp);

    size_t bytes = temp.GetDataSize();
    size_t rows = temp.GetRows();
    size_t cols = temp.GetCols();
    size_t stride = temp.GetStride();

    frame_sink_.bind(frame_sink_address_, bytes);

    // TODO: Mono case?
    shared_frame_ = frame_sink_.retrieve(rows, cols, CV_8UC3);
    shared_frame_.sample().set_rate_hz(frames_per_second_);

    // Use the shared_frame_.data, which points to a block of shared memory as
    // rbg_image's data buffer. When changes are made to rgb_image_, this is
    // automatically propagated into shmem and 'converted' into a cv::Mat
    // (although this 'conversion' is simply filling in appropriate header info,
    // which was accomplished in the call to frame_sink_.retrieve())
    rgb_image_ = std::make_unique<pg::Image>
            (rows, cols, stride, shared_frame_.data, bytes, pg::PIXEL_FORMAT_BGR);
}

bool PGGigECam::process() {

    int rc = grabImage();

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

        raw_image_.Convert(pg::PIXEL_FORMAT_BGR, rgb_image_.get());
        shared_frame_.sample().incrementCount(tick_);

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

    } while (i++ < rc);

    return false;
}

size_t PGGigECam::findNumCameras(void) {

    pg::Error error;
    pg::BusManager busMgr;

    unsigned int num_cameras = 0;

    error = busMgr.GetNumOfCameras(&num_cameras);
    if (error != pg::PGRERROR_OK)
        throw (rte(error.GetDescription()));

    return static_cast<size_t>(num_cameras);
}

void PGGigECam::printError(pg::Error error) {
    error.PrintErrorTrace();
    if (!camera_.IsConnected())
        std::cerr << "Camera must be connected before getting its info.\n";
}

bool PGGigECam::pollForTriggerReady() {

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

int PGGigECam::printBusInfo(void) {

    std::cout << "\n";
    std::cout << "*** BUS INFORMATION ***\n";
    std::cout << "No. cameras detected on bus: " << findNumCameras() << "\n";
    return 0;
}

int PGGigECam::printCameraInfo(void) {

    pg::CameraInfo camera_info;
    pg::Error error = camera_.GetCameraInfo(&camera_info);
    if (error != pg::PGRERROR_OK) {
        throw (rte(error.GetDescription()));
    }

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

    return 0;
}

void PGGigECam::printStreamChannelInfo(pg::GigEStreamChannel *pStreamChannel) {

    std::ostringstream ipAddress;
    ipAddress
        << (unsigned int) pStreamChannel->destinationIpAddress.octets[0] << "."
        << (unsigned int) pStreamChannel->destinationIpAddress.octets[1] << "."
        << (unsigned int) pStreamChannel->destinationIpAddress.octets[2] << "."
        << (unsigned int) pStreamChannel->destinationIpAddress.octets[3];

    std::cout << "Network interface: " << pStreamChannel->networkInterfaceIndex << "\n";
    //std::cout << "Host Port: " << pStreamChannel->hostPort << "\n";
    std::cout << "Do not fragment bit: " << (pStreamChannel->doNotFragment ? "Enabled" : "Disabled") << "\n";
    std::cout << "Packet size: " << pStreamChannel->packetSize << "\n";
    std::cout << "Inter packet delay: " << pStreamChannel->interPacketDelay << "\n";
    std::cout << "Destination IP address: " << ipAddress.str() << "\n";
    std::cout << "Source port (on camera): " << pStreamChannel->sourcePort << "\n\n";

}

void PGGigECam::fireSoftwareTrigger() {
    // Check that the trigger is ready

    if (use_software_trigger_) {

        pollForTriggerReady();

        std::cout << "Press the Enter key to initiate a software trigger\n";
        std::cin.ignore();

        // Fire software trigger
        const unsigned int k_softwareTrigger = 0x62C;
        const unsigned int k_fireVal = 0x80000000;
        pg::Error error;

        error = camera_.WriteRegister(k_softwareTrigger, k_fireVal);
        if (error != pg::PGRERROR_OK) {
            printError(error);
            std::cout << "Error firing software trigger\n";
        }

    } else {
        std::cout << "Cannot firing software trigger because software trigger has not been configured.\n";
    }
}

} /* namespace oat */
