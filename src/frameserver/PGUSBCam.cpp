//******************************************************************************
//* File:   PGUSBCam.cpp
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

#include "PGUSBCam.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include <cpptoml.h>
#include "FlyCapture2.h"
#include <opencv2/opencv.hpp>

#include "../../lib/utility/make_unique.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

PGUSBCam::PGUSBCam(const std::string &frame_sink_address,
                     const size_t index,
                     const double fps) :
  FrameServer(frame_sink_address)
, index_(index)
, frames_per_second_(fps)
{
    // Find the number of cameras on the bus
    findNumCameras();

    // Intialize frame timing
    tick_ = oat::Sample::Microseconds(0);
    tock_ = oat::Sample::Microseconds(0);
}

PGUSBCam::~PGUSBCam() {

    // Don't acquire error data in order to throw exception. Unsafe in
    // destructor
    camera_.StopCapture();
    camera_.Disconnect();
}

/**
 * Set default camera_ configuration
 */
void PGUSBCam::configure() {

    setCameraIndex(index_);
    connectToCamera();
    turnCameraOn();
    //setupStreamChannels();
    setupFrameRate(frames_per_second_, true);
    //setupExposure(true);
    setupShutter(true);
    setupGain(true);
    //setupWhiteBalance(false);
    //setupDefaultImageFormat();
    setupImageFormat();
    setupTrigger();
    //setupCameraFrameBuffer();
    setupEmbeddedImageData();
}

/**
 * Configure file using TOML configuration file
 * @param config_file The configuration file.
 * @param key The configuration key specifying options for this instance.
 */
void PGUSBCam::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options { "index",
                                       "fps",
                                       //"exposure",
                                       "shutter",
                                       "gain",
                                       "white_bal",
                                       "roi",
                                       "x_bin",
                                       "y_bin",
                                       "trigger_on",
                                       "trigger_rising",
                                       "trigger_mode",
                                       "trigger_pin",
                                       "enforce_fps",
                                       "strobe_pin",
                                       "calibration_file" };

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera_ configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Camera index
        {
            int64_t val;
            if (oat::config::getValue(this_config, "index", val, (int64_t)0, max_index_)) {
                index_ = val;
            }
        }

        setCameraIndex(index_);
        connectToCamera();
        turnCameraOn();
        //setupStreamChannels();

        // Frame rate
        if (oat::config::getValue(this_config, "fps", frames_per_second_, 0.0))
            setupFrameRate(frames_per_second_, false);
        else
            setupFrameRate(frames_per_second_, true);

//        // Set the exposure
//        {
//            double val;
//            if (oat::config::getValue(this_config, "exposure", val)) {
//                exposure_EV_ = val;
//                setupExposure(false);
//            } else {
//                setupExposure(true);
//            }
//        }

        // Set the shutter time
        {
            double val;
            if (!this_config->contains("exposure") &&
                oat::config::getValue(this_config, "shutter", val, 0.0, 1000.0)) {
                shutter_ms_ = val;
                setupShutter(false);
            } else {
                setupShutter(true);
            }
        }

        // Set the gain
        {
            double val;
            if (!this_config->contains("exposure") &&
                oat::config::getValue(this_config, "gain", val)) {
                gain_db_ = val;
                setupGain(false);
            } else {
                setupGain(true);
            }
        }

        // Set white balance
        {
            oat::config::Table wb;
            if (oat::config::getTable(this_config, "white_bal", wb)) {
                oat::config::getValue(wb, "red", white_bal_red_, (int64_t)0, (int64_t)1000, true);
                oat::config::getValue(wb, "blue", white_bal_blue_ , (int64_t)0, (int64_t)1000, true);
            } else {
                setupWhiteBalance(false);
            }
        }

        // Pixel binning
        {
            int64_t val;
            if (oat::config::getValue(this_config, "x_bin", val, (int64_t)0, (int64_t)8)) {
                x_bin_ = val;
            }

            if (oat::config::getValue(this_config, "y_bin", val, (int64_t)0, (int64_t)8)) {
                y_bin_ = val;
            }
        }

        // TODO: Must come after setting up image?
        //setupPixelBinning(x_bin_, y_bin_);

        // Set the ROI
        // TODO: Use the base class's included region_of_interest_ property instead of frame_offset
        // and frame_size
        {
            oat::config::Table roi;
            if (oat::config::getTable(this_config, "roi", roi)) {

                int64_t val;
                oat::config::getValue(roi, "x_offset", val, (int64_t)0, true);
                region_of_interest_.x = val;
                oat::config::getValue(roi, "y_offset", val, (int64_t)0, true);
                region_of_interest_.y = val;
                oat::config::getValue(roi, "width", val, (int64_t)0, true);
                region_of_interest_.width = val;
                oat::config::getValue(roi, "height", val, (int64_t)0, true);
                region_of_interest_.height = val;

            } else {
               // setupDefaultImageFormat();
            }
             setupImageFormat();
        }

        // Setup trigger
        if (oat::config::getValue(this_config, "trigger_on", use_trigger_)) {

            oat::config::getValue(this_config, "trigger_rising", trigger_polarity_);

            if (!oat::config::getValue(this_config, "trigger_mode_", trigger_mode_))
                trigger_mode_ = 14;

            if (trigger_mode_ == 7)
                use_software_trigger_ = true;
            else
                use_software_trigger_ = false;

            if (!oat::config::getValue(this_config,
                                       "trigger_source",
                                        trigger_source_pin_,
                                        (int64_t)0,
                                        (int64_t)1)) {
                trigger_source_pin_ = 0;
            }
        }

        if (!oat::config::getValue(this_config,
                                   "strobe_pin",
                                    strobe_output_pin_,
                                    (int64_t)0,
                                    (int64_t)1)) {

            if (trigger_source_pin_)
                strobe_output_pin_ = 0;
            else
                strobe_output_pin_ = 1;
        }


        if (trigger_source_pin_ == strobe_output_pin_)
            throw std::runtime_error("Stobe pin must be different from trigger pin.");

        setupTrigger();

//        if (this_config->contains("retry")) {
//
//            num_transmit_retries_ = *this_config->get_as<int64_t>("retry");
//
//            if (num_transmit_retries_ > 0) {
//                use_frame_buffer_ = true;
//            } else {
//                use_frame_buffer_ = false;
//            }
//        }
//
//        setupCameraFrameBuffer();

        // Should the FPS setting be enforced by retransmitting frames in the
        // case of dropped triggers
        oat::config::getValue(this_config, "enforce_fps", enforce_fps_);

        // Embed timestamp with frames
        setupEmbeddedImageData();

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }

}

int PGUSBCam::setCameraIndex(unsigned int requested_idx) {

    // Print bus information and find the number of cameras on the bus
    printBusInfo();

    if (num_cameras_ > requested_idx) {
        index_ = requested_idx;
    } else {
        throw (std::runtime_error("Requested camera index " +
                std::to_string(requested_idx) + " is out of range.\n"));
    }

    return 0;
}

int PGUSBCam::connectToCamera(void) {

    std::cout << "Connecting to camera: " << index_ << "\n";

    pg::BusManager busMgr;
    pg::PGRGuid guid;
    pg::Error error = busMgr.GetCameraFromIndex(index_, &guid);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    error = camera_.Connect(&guid);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    printCameraInfo();

    std::cout << "Restoring default camcera acqusition settings...\n";

    error = camera_.RestoreFromMemoryChannel(0);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    std::cout << "Default settings restored.\n";

    return 0;
}

// int PGUSBCam::setupStreamChannels() {

//     unsigned int numStreamChannels = 0;
//     pg::Error error = camera_.GetNumStreamChannels(&numStreamChannels);
//     if (error != pg::PGRERROR_OK) {
//         throw (std::runtime_error(error.GetDescription()));
//     }

//     for (unsigned int i = 0; i < numStreamChannels; i++) {
//         pg::GigEStreamChannel streamChannel;
//         error = camera_.GetGigEStreamChannelInfo(i, &streamChannel);
//         if (error != pg::PGRERROR_OK) {
//             throw (std::runtime_error(error.GetDescription()));
//         }

//         // TODO: This is not going to be valid if cameras are on a switch...
//         streamChannel.destinationIpAddress.octets[0] = 224;
//         streamChannel.destinationIpAddress.octets[1] = 0;
//         streamChannel.destinationIpAddress.octets[2] = 0;
//         streamChannel.destinationIpAddress.octets[3] = 1;

//         // TODO: Make a more reasoned choice for these parameters based on the
//         // number of cameras on the system...
//         streamChannel.packetSize = 9000;
//         streamChannel.interPacketDelay = 250;

//         error = camera_.SetGigEStreamChannelInfo(i, &streamChannel);
//         if (error != pg::PGRERROR_OK) {
//             throw (std::runtime_error(error.GetDescription()));
//         }

//         std::cout << "Printing stream channel information for channel " << i << "\n";
//         printStreamChannelInfo(&streamChannel);
//     }

//     return 0;
// }

int PGUSBCam::setupFrameRate(double fps, bool is_auto) {

    std::cout << "Setting up frame rate...\n";

    pg::Property prop;
    prop.type = pg::FRAME_RATE;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    prop.autoManualMode = is_auto;

    if (!is_auto) {
        prop.absValue = fps;
        std::cout << "Frame rate set to " + std::to_string(prop.absValue) + " FPS.\n";
    } else {
        std::cout << "Frame rate set to auto.\n";
    }

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    // If set to auto, then get the automatically configured frame frame rate
    if (is_auto) {
        error = camera_.GetProperty(&prop);
        if (error != pg::PGRERROR_OK) {
            throw (std::runtime_error(error.GetDescription()));
        }

        frames_per_second_ = prop.absValue;
    }

    return 0;
}

int PGUSBCam::setupShutter(bool is_auto) {

    std::cout << "Setting up shutter...\n";

    pg::Property prop;
    prop.type = pg::SHUTTER;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = shutter_ms_;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    if (is_auto) {
        std::cout << "Shutter set to auto.\n";
    } else {
        std::cout << "Shutter time set to "
                  << std::fixed
                  << std::setprecision(2)
                  << shutter_ms_ << " ms.\n";
    }

    return 0;
}

int PGUSBCam::setupShutter(float shutter_ms_in) {

    shutter_ms_ = shutter_ms_in;
    setupShutter(false);

    return 0;
}

int PGUSBCam::setupGain(bool is_auto) {

    std::cout << "Setting camera gain...\n";

    pg::Property prop;
    prop.type = pg::GAIN;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = gain_db_;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    if (is_auto) {
        std::cout << "Gain set to auto.\n";
    } else {
        std::cout << "Gain set to " << std::fixed << std::setprecision(2) << gain_db_ << " dB.\n";
    }

    return 0;
}

int PGUSBCam::setupGain(float gain_dB_in) {

    gain_db_ = gain_dB_in;
    setupGain(false);

    return 0;
}

int PGUSBCam::setupExposure(bool is_auto) {

    std::cout << "Setting up exposure...\n";

    setupShutter(true);
    setupGain(true);
    pg::Property prop;
    prop.type = pg::AUTO_EXPOSURE;
    pg::Error error = camera_.GetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    prop.onOff = true;
    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = exposure_EV_;

    error = camera_.SetProperty(&prop);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    if (is_auto) {
        std::cout << "Exposure set to auto.\n";
    } else {
        std::cout << "Exposure set to "
                  << std::fixed
                  << std::setprecision(2)
                  << exposure_EV_ << " EV.\n";
    }

    return 0;
}

int PGUSBCam::setupExposure(float exposure_EV_in) {

    exposure_EV_ = exposure_EV_in;
    setupExposure(false);

    return 0;
}

int PGUSBCam::setupWhiteBalance(bool is_on) {

    std::cout << "Setting camera white balance...\n";

    pg::Property prop;
    prop.type = pg::WHITE_BALANCE;
    pg::Error error = camera_.GetProperty(&prop);

    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    prop.onOff = is_on;
    prop.autoManualMode = false;
    prop.absControl = false;
    prop.valueA = white_bal_red_;
    prop.valueB = white_bal_blue_;

    error = camera_.SetProperty(&prop);

    if(error == pg::PGRERROR_PROPERTY_NOT_PRESENT)
    {
        std::cout  << "White balance proprty can't be set on this camera: " << error.GetDescription() << "\n";
    } else {

        if (error != pg::PGRERROR_OK) {
            throw (std::runtime_error(error.GetDescription()));
        }

        if (is_on) {
            std::cout << "White balance set to: \n";
            std::cout << "\tRed: "
                      << std::fixed
                      << std::setprecision(2)
                      << white_bal_red_ << "\n";
            std::cout << "\tBlue: "
                      << std::fixed
                      << std::setprecision(2)
                      << white_bal_blue_ << "\n";
        } else {
            std::cout << "White balance turned off.\n";
        }
    }
    return 0;
}

int PGUSBCam::setupWhiteBalance(int white_bal_red_in, int white_bal_blue_in) {

    white_bal_red_ = white_bal_red_in;
    white_bal_blue_ = white_bal_blue_in;
    setupWhiteBalance(true);

    return 0;
}

/**
 * Default image setup. Image uses all available pixels and BRG pixel format.
 *
 * @return 0 if successful.
 */
// int PGUSBCam::setupDefaultImageFormat() {

//     std::cout << "Querying GigE image setting information...\n";

//     pg::Format7ImageSettingsInfo image_settings_info;
//     pg::Error error = camera_.GetFormat7ImageSettingsInfo(&image_settings_info);
//     if (error != pg::PGRERROR_OK) {
//         printError(error);
//         return -1;
//     }

//     region_of_interest_.x = 0;
//     region_of_interest_.y = 0;
//     region_of_interest_.width = image_settings_info.maxWidth;
//     region_of_interest_.height = image_settings_info.maxHeight;

//     pg::Format7ImageSettings imageSettings;
//     imageSettings.offsetX = region_of_interest_.x;
//     imageSettings.offsetY = region_of_interest_.y;
//     imageSettings.height = region_of_interest_.height;
//     imageSettings.width = region_of_interest_.width;
//     imageSettings.pixelFormat = pg::PIXEL_FORMAT_RAW12;

//     std::cout << "Setting GigE image settings...\n";

//     error = camera_.SetFormat7ImageSettings(&imageSettings);
//     if (error != pg::PGRERROR_OK) {
//         throw (std::runtime_error(error.GetDescription()));
//     }

//     return 0;

// }

/**
 * Custom image setup. Image uses the internally specified ROI settings.
 * @return
 */
int PGUSBCam::setupImageFormat() {

    std::cout << "Querying image settings information...\n";


// Query for available Format 7 modes

    const pg::Mode k_fmt7Mode = pg::MODE_0;
    const pg::PixelFormat k_fmt7PixFmt = pg::PIXEL_FORMAT_MONO8;


    pg::Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = k_fmt7Mode;
    pg::Error error = camera_.GetFormat7Info( &fmt7Info, &supported );
    if (error != pg::PGRERROR_OK)
    {
         throw (std::runtime_error(error.GetDescription()));
        //PrintError( error );
        //return -1;
    }

    //PrintFormat7Capabilities( fmt7Info );

    if ( (k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0 )
    {
        // Pixel format not supported!
        std::cout << "Pixel format is not supported" << std::endl;
         throw (std::runtime_error(error.GetDescription()));
        //return -1;
    }


    // pg::Format7ImageSettings image_settings_info;
    // error = camera_.GetFormat7ImageSettingsInfo(&image_settings_info);
    // if (error != pg::PGRERROR_OK) {
    //     throw (std::runtime_error(error.GetDescription()));
    // }

    // if (region_of_interest_.x > fmt7Info.maxWidth ||
    //         region_of_interest_.y > fmt7Info.maxHeight) {
    //     throw (std::runtime_error("ROI pixel offsets are larger than the sensor array. Exiting.\n"));
    // }

    // if ((region_of_interest_.width + region_of_interest_.x) > fmt7Info.maxWidth) {
    //     throw (std::runtime_error("Current X-axis ROI settings are off the sensor array\n"));
    // }

    // if ((region_of_interest_.height + region_of_interest_.y) > fmt7Info.maxHeight) {
    //     throw (std::runtime_error("Current Y-axis ROI settings are off the sensor array\n"));
    // }


    pg::Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = fmt7Info.mode;
    fmt7ImageSettings.offsetX = 0;
    fmt7ImageSettings.offsetY = 0;
    fmt7ImageSettings.width = 640;
    fmt7ImageSettings.height = 480;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;



    std::cout << "Setting image settings...\n";

    // error = camera_.SetFormat7ImageSettings(&imageSettings);
    // if (error != pg::PGRERROR_OK) {
    //     throw (std::runtime_error(error.GetDescription()));
    // }

    pg::Format7PacketInfo fmt7PacketInfo;
    bool valid = true;

    // Validate the settings to make sure that they are valid
    error = camera_.ValidateFormat7Settings(
        &fmt7ImageSettings,
        &valid,
        &fmt7PacketInfo );
    if (error != pg::PGRERROR_OK)
    {
        //PrintError( error );
        std::cout << "error validating format 7 format\n";
         throw ( std::runtime_error(error.GetDescription()));

        //return -1;
    }

    if ( !valid )
    {
        // Settings are not valid
        //cout << "Format7 settings are not valid" << endl;
        throw (std::runtime_error("oh noes something is not valid\n"));
        //return -1;
    }

    // Set the settings to the camera
    error = camera_.SetFormat7Configuration(
        &fmt7ImageSettings,
        fmt7PacketInfo.recommendedBytesPerPacket );
    if (error != pg::PGRERROR_OK)
    {
        //PrintError( error );
        throw (std::runtime_error(error.GetDescription()));
        //return -1;
    }

    return 0;
}

// int PGUSBCam::setupPixelBinning(int x_bin, int y_bin) {

//     std::cout << "Setting image binning...\n";

//     // On-board image binning
//     pg::Error error;
//     error = camera_.SetGigEImageBinningSettings(x_bin, y_bin);
//     if (error != pg::PGRERROR_OK) {
//         throw (std::runtime_error(error.GetDescription()));
//     }

//     std::cout << "Onboard binning set to [" << x_bin << " " << y_bin << "]\n";

//     return 0;
// }

//int PGUSBCam::setupCameraFrameBuffer() {
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
//            throw (std::runtime_error(error.GetDescription()));
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
int PGUSBCam::turnCameraOn() {

    // Power on the camera_
    const unsigned int k_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    pg::Error error = camera_.WriteRegister(k_cameraPower, k_powerVal);
    if (error != pg::PGRERROR_OK && error != pg::PGRERROR_NOT_IMPLEMENTED){
        throw (std::runtime_error(error.GetDescription()));
    }

    const unsigned int millisecondsToSleep = 100;
    unsigned int regVal = 0;
    unsigned int retries = 10;

    // Wait for camera_ to complete power-up
    do {
#if defined(WIN32) || defined(WIN64)
        Sleep(millisecondsToSleep);
#else
        usleep(millisecondsToSleep * 1000);
#endif
        error = camera_.ReadRegister(k_cameraPower, &regVal);
        if (error == pg::PGRERROR_TIMEOUT) {
            // ignore timeout errors, camera_ may not be responding to
            // register reads during power-up
        } else if (error != pg::PGRERROR_OK) {
            throw (std::runtime_error(error.GetDescription()));
        }

        retries--;
    } while ((regVal & k_powerVal) == 0 && retries > 0);

    // Check for timeout errors after retrying
    if (error == pg::PGRERROR_TIMEOUT) {
        throw (std::runtime_error(error.GetDescription()));
    }

    return 0;
}

int PGUSBCam::setupTrigger() {

    // Get current trigger settings
    pg::TriggerModeInfo trigger_mode_info;
    pg::Error error = camera_.GetTriggerModeInfo(&trigger_mode_info);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    if (trigger_mode_info.present != true) {
        throw (std::runtime_error(error.GetDescription()));
    }

    pg::TriggerMode triggerMode;
    error = camera_.GetTriggerMode(&triggerMode);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    triggerMode.onOff = use_trigger_;
    triggerMode.polarity = trigger_polarity_;
    triggerMode.mode = trigger_mode_;
    triggerMode.parameter = 0;
    triggerMode.source = trigger_source_pin_;

    error = camera_.SetTriggerMode(&triggerMode);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    // Use GPIO as strobe to indicate shutter open/close
    pg::StrobeControl strobe;
    strobe.source = strobe_output_pin_;
    strobe.onOff = true;
    strobe.polarity = 1;
    strobe.delay = 0.0f;
    strobe.duration = 0.0f;

    error = camera_.SetStrobe(&strobe);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    // Setup frame buffering
    // NOTE: For some reason grabMode = pg::BUFFER_FRAMES does not play nicely
    // with the time-stamp correction for dropped triggers. I have yet to
    // understand why.
    pg::FC2Config flyCapConfig;
    error = camera_.GetConfiguration(&flyCapConfig);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    flyCapConfig.grabTimeout = 10;
    flyCapConfig.grabMode = pg::DROP_FRAMES;
    flyCapConfig.highPerformanceRetrieveBuffer = true;
    //flyCapConfig.numBuffers = 1;

    error = camera_.SetConfiguration(&flyCapConfig);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

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

    // Camera is ready, start capturing images
    error = camera_.StartCapture();
    if (error == pg::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
        throw (std::runtime_error("Interface bandwidth exceeded. Cannot start camera_..\n"));
    } else if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    acquisition_started_ = true;
    if (use_trigger_ && use_software_trigger_) {
        std::cout << "Trigger the camera by pressing Enter\n";
    } else if (use_trigger_) {
        std::cout << "Trigger the camera by sending a trigger pulse to GPIO_"
                  << triggerMode.source << "\n";
    } else {
        std::cout << "Camera by started in free running mode.\n";
    }

    return 0;
}

int PGUSBCam::setupEmbeddedImageData() {

    pg::EmbeddedImageInfo embeddedInfo;
    pg::Error error = camera_.GetEmbeddedImageInfo(&embeddedInfo);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    // For now, only inlcude timestamp
    embeddedInfo.timestamp.onOff = true;

    error = camera_.SetEmbeddedImageInfo(&embeddedInfo);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    // TODO: HACK! See https://github.com/jonnew/Oat/issues/11
    int i = 0;
    while (use_trigger_ && (error == pg::PGRERROR_OK || i < 10)) {
        error = camera_.RetrieveBuffer(&raw_image_);
        i++;
    }

    return 0;
}

// TODO: event driven acquisition.
//void PGUSBCam::onGrabbedImage(Image* pImage, const void* pCallbackData) {
//
//    &raw_image_ = pImage;
//    current_frame = imageToMat();
//    serveMat();
//
//}

// TODO: implement onboard buffer and perform retry a RetrieveBuffer
// A single time if a torn image is detected.
/**
 * Grab a frame from the camera_'s buffer.
 *
 * @return return code meaning:
 *   -1 : Grab timeout occurred
 *    0 : Successful grab
 *  >=1 : A return code greater than or equal to 1 indicates only occurs if a
 *        strict frame rate is enforced when using an external trigger. Because it is
 *        possible for PG cameras to skip triggers, the function will check the time
 *        between consecutive frames and indicate the estimated number of missed
 *        frames, if any, using this code.
 */
int PGUSBCam::grabImage() {

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
    double delay = (double)((tick_ - tock_).count()) / 1.0e6;;

    if (first_frame_ || !enforce_fps_) {
        first_frame_ = false;
        return 0;
    } else if (delay > 0.0) {
        // Return the number of skipped frames. This should be 0, but PG cameras
        // reject triggers on some ocations and we need to fill in the blanks to
        // prevent offsets...
        return (int)(std::round(frames_per_second_ * delay  - 1.0));
    } else {
        return 0;
    }
}

uint64_t PGUSBCam::uncycle1394Timestamp(int ieee_1394_sec,
                                         int ieee_1394_cycle) {

    if (ieee_1394_sec - last_ieee_1394_sec_ < 0)
        ieee_1394_cycle_index_++;

    last_ieee_1394_sec_ = ieee_1394_sec;

    return (uint64_t)(ieee_1394_cycle_index_ * 128 + ieee_1394_sec) * IEEE_1394_HZ
           + ieee_1394_cycle
           - ieee_1394_start_cycle_;
}

void PGUSBCam::connectToNode() {


    pg::Format7ImageSettings pImageSettings;
    unsigned int pPacketSize;
    float pPercentage;

    pg::Error error = camera_.GetFormat7Configuration( &pImageSettings, &pPacketSize, &pPercentage);

    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
    }

    pg::Image temp(pImageSettings.height,
                   pImageSettings.width,
                   pg::PIXEL_FORMAT_BGR);

    raw_image_.Convert(pg::PIXEL_FORMAT_BGR, &temp);

    std::cout <<  temp.GetDataSize() << " " << temp.GetRows() << " " << temp.GetCols() << " " << temp.GetStride() << std::endl;

    size_t bytes = temp.GetDataSize();
    size_t rows = temp.GetRows();
    size_t cols = temp.GetCols();
    size_t stride = temp.GetStride();

    // size_t bytes = 640*480*16;
    // size_t rows = 480;
    // size_t cols = 640;
    // size_t stride = 640;



    frame_sink_.bind(frame_sink_address_, bytes);

    shared_frame_ = frame_sink_.retrieve(rows, cols, CV_8UC3);
    internal_sample_.set_rate_hz(frames_per_second_);

    // Use the shared_frame_.data, which points to a block of shared memory as
    // rbg_image's data buffer. When changes are made to rgb_image_, this is
    // automatically propagated into shmem and 'converted' into a cv::Mat
    // (although this 'conversion' is simply filling in appropriate header info,
    // which was accomplished in the call to frame_sink_.retrieve())
    rgb_image_ = std::make_unique<pg::Image>
            (rows, cols, stride, shared_frame_.data, bytes, pg::PIXEL_FORMAT_BGR);
}

bool PGUSBCam::serveFrame() {

    int rc = grabImage();

    // There was a grab timeout.
    // Allow check to see if SIGINT occurred.
    if (rc == -1)
        return false;

//#ifndef NDEBUG
    if (rc > 0) {
        std::cerr << oat::Warn("Frame re-transmission due to " +
                               std::to_string(rc) +
                               " skipped trigger(s).\n");
    }
//#endif

    int i = 0;
    do {

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        frame_sink_.wait();

        raw_image_.Convert(pg::PIXEL_FORMAT_BGR, rgb_image_.get());
        shared_frame_.sample() = internal_sample_;

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        // Pure SINKs increment sample count
        internal_sample_.incrementCount(tick_);

    } while (i++ < rc);

    return false;
}

int PGUSBCam::findNumCameras(void) {

    pg::Error error;
    pg::BusManager busMgr;

    error = busMgr.GetNumOfCameras(&num_cameras_);
    if (num_cameras_ == 0)
        throw (std::runtime_error("No GigE cameras were detected.\n"));

    max_index_ = num_cameras_ - 1;
    if (error != pg::PGRERROR_OK)
        throw (std::runtime_error(error.GetDescription()));

    return 0;
}

void PGUSBCam::printError(pg::Error error) {
    error.PrintErrorTrace();
    if (!camera_.IsConnected())
        std::cerr << "Camera must be connected before getting its info.\n";
}

bool PGUSBCam::pollForTriggerReady() {

    const unsigned int k_softwareTrigger = 0x62C;
    pg::Error error;
    unsigned int regVal = 0;

    do {
        error = camera_.ReadRegister(k_softwareTrigger, &regVal);
        if (error != pg::PGRERROR_OK) {
            throw (std::runtime_error(error.GetDescription()));
        }

    } while ((regVal >> 31) != 0);

    return true;
}

int PGUSBCam::printBusInfo(void) {

    std::cout << "\n";
    std::cout << "*** BUS INFORMATION ***\n";
    std::cout << "No. cameras detected on bus: " << num_cameras_ << "\n";
    return 0;
}

int PGUSBCam::printCameraInfo(void) {

    pg::CameraInfo camera_info;
    pg::Error error = camera_.GetCameraInfo(&camera_info);
    if (error != pg::PGRERROR_OK) {
        throw (std::runtime_error(error.GetDescription()));
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
   // std::cout << "GigE version :"
   //           << camera_info.gigEMajorVersion << "."
   //           << camera_info.gigEMinorVersion << "\n";
    std::cout << "User defined name :" << camera_info.userDefinedName << "\n";
    //std::cout << "XML URL 1: " << camera_info.xmlURL1 << "\n";
    //std::cout << "XML URL 2: " << camera_info.xmlURL2 << "\n";
    std::cout << "MAC address: " << macAddress.str() << "\n";
    std::cout << "IP address: " << ipAddress.str() << "\n";
    std::cout << "Subnet mask: " << subnetMask.str() << "\n";
    std::cout << "Default gateway: " << defaultGateway.str() << "\n\n";

    return 0;
}

// void PGUSBCam::printStreamChannelInfo(pg::GigEStreamChannel *pStreamChannel) {
//     //char ipAddress[32];
//     std::ostringstream ipAddress;
//     ipAddress
//         << (unsigned int) pStreamChannel->destinationIpAddress.octets[0] << "."
//         << (unsigned int) pStreamChannel->destinationIpAddress.octets[1] << "."
//         << (unsigned int) pStreamChannel->destinationIpAddress.octets[2] << "."
//         << (unsigned int) pStreamChannel->destinationIpAddress.octets[3];

//     std::cout << "Network interface: " << pStreamChannel->networkInterfaceIndex << "\n";
//     //std::cout << "Host Port: " << pStreamChannel->hostPort << "\n";
//     std::cout << "Do not fragment bit: " << (pStreamChannel->doNotFragment ? "Enabled" : "Disabled") << "\n";
//     std::cout << "Packet size: " << pStreamChannel->packetSize << "\n";
//     std::cout << "Inter packet delay: " << pStreamChannel->interPacketDelay << "\n";
//     std::cout << "Destination IP address: " << ipAddress.str() << "\n";
//     std::cout << "Source port (on camera): " << pStreamChannel->sourcePort << "\n\n";

// }

void PGUSBCam::fireSoftwareTrigger() {
    // Check that the trigger is ready

    if (use_trigger_ && use_software_trigger_) {

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
