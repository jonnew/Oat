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

#include "PGGigECam.h"

#include <string>
#include <stdlib.h> 
#include <unistd.h> // TODO: Don't know how to include only if LINUX with cmake
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/core/core.hpp>

#include "FlyCapture2.h"

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/shmem/SharedCVMatHeader.h"
#include "../../lib/shmem/MatServer.h"

#include "stdafx.h"

using namespace FlyCapture2;

PGGigECam::PGGigECam(std::string frame_sink_name) :
Camera(frame_sink_name)
, num_cameras(0)
, index(0)
, shutter_ms(0)
, gain_dB(0)
, exposure_EV(0)
, aquisition_started(false)
, use_trigger(false)
, use_software_trigger(false)
, trigger_polarity(true)
, trigger_mode(14)
, trigger_source_pin(0) {

    // Initialize the frame size
    frame_size = cv::Size(728, 728);
    frame_offset = cv::Size(0, 0);

}

/**
 * Set default camera configuration
 */
void PGGigECam::configure() {

    name = "Default";
    setCameraIndex(0);
    connectToCamera();
    turnCameraOn();
    setupStreamChannels();
    setupExposure(true);
    setupShutter(true);
    setupGain(true);
    setupWhiteBalance(false);
    setupDefaultImageFormat();
    setupTrigger();
}

/**
 * Configure file using TOML configuration file
 * @param config_file The configuration file.
 * @param key The configuration key specifying options for this instance.
 */
void PGGigECam::configure(std::string config_file, std::string key) {

    cpptoml::table config;

    try {
        config = cpptoml::parse_file(config_file);

        //std::cout << "Parsed the following configuration...\n" << "\n";
        //std::cout << config << "\n";
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << "\n";
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(key)) {

            auto camera_config = *config.get_table(key);
            name = key;

            // Set the camera index
            if (camera_config.contains("index"))
                index = (unsigned int) (*camera_config.get_as<int64_t>("index"));
            else
                setCameraIndex(0);

            connectToCamera();
            turnCameraOn();
            setupStreamChannels();

            // Set the exposure
            if (camera_config.contains("exposure")) {
                exposure_EV = (float) (*camera_config.get_as<double>("exposure"));
                setupExposure(false);
            } else {
                // Default to auto exposure
                setupExposure(true);
            }

            // Set the shutter time
            if (camera_config.contains("shutter") && !camera_config.contains("exposure")) {
                shutter_ms = (float) (*camera_config.get_as<double>("shutter"));
                setupShutter(false);
            } else {
                // Default to auto shutter
                setupShutter(true);
            }

            // Set the gain
            if (camera_config.contains("gain")&& !camera_config.contains("exposure")) {
                gain_dB = (float) (*camera_config.get_as<double>("gain"));
                setupGain(false);
            } else {
                // Default to auto gain
                setupGain(true);
            }

            // Set white balance
            if (camera_config.contains("white_bal")) {

                auto wb = *camera_config.get_table("white_bal");

                white_bal_red = (int) (*wb.get_as<int64_t>("red"));
                white_bal_blue = (int) (*wb.get_as<int64_t>("blue"));
                setupWhiteBalance(true);
            } else {

                // Default: turn white balance off
                setupWhiteBalance(false);
            }

            // Set the ROI
            if (camera_config.contains("roi")) {

                auto roi = *camera_config.get_table("roi");

                frame_offset.width = (int) (*roi.get_as<int64_t>("x_offset"));
                frame_offset.height = (int) (*roi.get_as<int64_t>("y_offset"));
                frame_size.width = (int) (*roi.get_as<int64_t>("width"));
                frame_size.height = (int) (*roi.get_as<int64_t>("height"));
                setupImageFormat();
            } else {
                setupDefaultImageFormat();
            }

            if (camera_config.contains("trigger_on")) {

                use_trigger = *camera_config.get_as<bool>("trigger_on");

                if (use_trigger) {

                    if (camera_config.contains("trigger_polarity")) {
                        trigger_polarity = *camera_config.get_as<int64_t>("trigger_polarity");
                    } else {
                        trigger_polarity = true;
                    }

                    if (camera_config.contains("trigger_mode")) {
                        trigger_mode = *camera_config.get_as<int64_t>("trigger_mode");
                    } else {
                        trigger_mode = 14;
                    }
                    
                    if (trigger_mode == 7)
                        use_software_trigger = true;
                    else
                        use_software_trigger = false;

                    if (camera_config.contains("trigger_source")) {
                        trigger_source_pin = *camera_config.get_as<int64_t>("trigger_source");
                    } else {
                        trigger_source_pin = 0;
                    }
                }
            } else {
                use_trigger = false;
            }
            setupTrigger();

        } else {
            std::cerr << "No camera configuration named \"" + key + "\" was provided. Exiting.\n";
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}

int PGGigECam::setCameraIndex(unsigned int requestedIdx) {

    // Find the number of cameras on the bus
    findNumCameras();

    // Print bus information and find the number of cameras on the bus
    printBusInfo();

    if (num_cameras > 0) {
        index = requestedIdx;

    } else {
        std::cerr << "No cameras were detected, so setting an index is not possible.\n";
        exit(EXIT_FAILURE);
    }

    return 0;
}

int PGGigECam::connectToCamera(void) {

    std::cout << "Connecting to camera: " << index << "\n";

    BusManager busMgr;
    PGRGuid guid;
    Error error = busMgr.GetCameraFromIndex(index, &guid);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    // Connect to a camera
    error = camera.Connect(&guid);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    // Print camera information
    printCameraInfo();

    return 0;
}

int PGGigECam::setupStreamChannels() {

    unsigned int numStreamChannels = 0;
    Error error = camera.GetNumStreamChannels(&numStreamChannels);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < numStreamChannels; i++) {
        GigEStreamChannel streamChannel;
        error = camera.GetGigEStreamChannelInfo(i, &streamChannel);
        if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
        }

        // TODO: This is not going to valid if cameras are on a switch...
        streamChannel.destinationIpAddress.octets[0] = 224;
        streamChannel.destinationIpAddress.octets[1] = 0;
        streamChannel.destinationIpAddress.octets[2] = 0;
        streamChannel.destinationIpAddress.octets[3] = 1;

        // TODO: Make a more reasoned choice for these parameters based on the
        // number of cameras on the system...
        streamChannel.packetSize = 9000;
        streamChannel.interPacketDelay = 1000;

        error = camera.SetGigEStreamChannelInfo(i, &streamChannel);
        if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
        }

        std::cout << "Printing stream channel information for channel " << i << "\n";
        printStreamChannelInfo(&streamChannel);
    }

    return 0;
}

int PGGigECam::setupShutter(bool is_auto) {

    std::cout << "Setting up shutter...\n";

    Property prop;
    prop.type = SHUTTER;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = shutter_ms;

    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (is_auto) {
        std::cout << "Shutter set to auto.\n";
    } else {
        std::cout << "Shutter time set to " << std::fixed << std::setprecision(2) << shutter_ms << " ms.\n";
    }

    return 0;
}

int PGGigECam::setupShutter(float shutter_ms_in) {

    shutter_ms = shutter_ms_in;
    setupShutter(false);
}

int PGGigECam::setupGain(bool is_auto) {

    std::cout << "Setting camera gain...\n";

    Property prop;
    prop.type = GAIN;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = gain_dB;

    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (is_auto) {
        std::cout << "Gain set to auto.\n";
    } else {
        std::cout << "Gain set to " << std::fixed << std::setprecision(2) << gain_dB << " dB.\n";
    }

    return 0;
}

int PGGigECam::setupGain(float gain_dB_in) {

    gain_dB = gain_dB_in;
    setupGain(false);

}

int PGGigECam::setupExposure(bool is_auto) {

    std::cout << "Setting up exposure...\n";

    setupShutter(true);
    setupGain(true);
    Property prop;
    prop.type = AUTO_EXPOSURE;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    prop.onOff = true;
    prop.autoManualMode = is_auto;
    prop.absControl = true;
    prop.absValue = exposure_EV;

    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (is_auto) {
        std::cout << "Exposure set to auto.\n";
    } else {
        std::cout << "Exposure set to " << std::fixed << std::setprecision(2) << exposure_EV << " EV.\n";
    }

    return 0;
}

int PGGigECam::setupExposure(float exposure_EV_in) {

    exposure_EV = exposure_EV_in;
    setupExposure(false);
}

int PGGigECam::setupWhiteBalance(bool is_on) {

    std::cout << "Setting camera white balance...\n";

    Property prop;
    prop.type = WHITE_BALANCE;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    prop.onOff = is_on;
    prop.autoManualMode = false;
    prop.absControl = false;
    prop.valueA = white_bal_red;
    prop.valueB = white_bal_blue;

    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (is_on) {
        std::cout << "White balance set to: \n";
        std::cout << "\tRed: " << std::fixed << std::setprecision(2) << white_bal_red << "\n";
        std::cout << "\tBlue: " << std::fixed << std::setprecision(2) << white_bal_blue << "\n";
    } else {
        std::cout << "White balance turned off.\n";
    }

    return 0;
}

int PGGigECam::setupWhiteBalance(int white_bal_red_in, int white_bal_blue_in) {

    white_bal_red = white_bal_red_in;
    white_bal_blue = white_bal_blue_in;
    setupWhiteBalance(true);

}

/**
 * Default image setup. Image uses all available pixels and BRG pixel format. 
 * 
 * @return 0 if successful.
 */
int PGGigECam::setupDefaultImageFormat() {

    std::cout << "Querying GigE image setting information...\n";

    Error error = camera.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != PGRERROR_OK) {
        printError(error);
        return -1;
    }

    frame_offset.width = 0;
    frame_offset.height = 0;
    frame_size.width = image_settings_info.maxWidth;
    frame_size.height = image_settings_info.maxHeight;

    GigEImageSettings imageSettings;
    imageSettings.offsetX = frame_offset.width;
    imageSettings.offsetY = frame_offset.height;
    imageSettings.height = frame_size.height;
    imageSettings.width = frame_size.width;
    imageSettings.pixelFormat = PIXEL_FORMAT_RAW12;
    //imageSettings.pixelFormat = PIXEL_FORMAT_MONO8;

    std::cout << "Setting GigE image settings...\n";

    error = camera.SetGigEImageSettings(&imageSettings);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    return 0;

}

/**
 * Custom image setup. Image uses the internally specified ROI settings.
 * @return 
 */
int PGGigECam::setupImageFormat() {

    std::cout << "Querying GigE image setting information...\n";

    Error error = camera.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (frame_offset.width > image_settings_info.maxWidth ||
            frame_offset.height > image_settings_info.maxHeight) {

        std::cerr << "ROI pixel offsets are larger than the CCD array. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    if ((frame_offset.width + frame_size.width) > image_settings_info.maxWidth) {

        frame_size.width = image_settings_info.maxWidth - frame_offset.width;
        std::cout << "WARNING: Current X-axis ROI settings is off the CCD array\n";
        std::cout << "WARNING: Cropping the ROI to fit on the array: \n";
        std::cout << "\tFrame width: " + frame_size.width << "\n";
        std::cout << "\tFrame offset: " + frame_offset.width << "\n";
    }

    if ((frame_offset.height + frame_size.height) > image_settings_info.maxHeight) {

        frame_size.height = image_settings_info.maxHeight - frame_offset.height;
        std::cout << "WARNING: Current Y-axis ROI settings is off the CCD array\n";
        std::cout << "WARNING: Cropping the ROI to fit on the array: \n";
        std::cout << "\tFrame height: " + frame_size.height << "\n";
        std::cout << "\tFrame offset: " + frame_offset.height << "\n";
    }

    GigEImageSettings imageSettings;
    imageSettings.offsetX = frame_offset.width;
    imageSettings.offsetY = frame_offset.height;
    imageSettings.height = frame_size.height;
    imageSettings.width = frame_size.width;
    imageSettings.pixelFormat = PIXEL_FORMAT_RAW12;

    std::cout << "Setting GigE image settings...\n";

    error = camera.SetGigEImageSettings(&imageSettings);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

/**
 * Once connected to the camera, issue power on command.
 * 
 * @return 0 if successful.
 */
int PGGigECam::turnCameraOn() {

    // Power on the camera
    const unsigned int k_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    Error error = camera.WriteRegister(k_cameraPower, k_powerVal);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    const unsigned int millisecondsToSleep = 100;
    unsigned int regVal = 0;
    unsigned int retries = 10;

    // Wait for camera to complete power-up
    do {
#if defined(WIN32) || defined(WIN64)
        Sleep(millisecondsToSleep);
#else
        usleep(millisecondsToSleep * 1000);
#endif
        error = camera.ReadRegister(k_cameraPower, &regVal);
        if (error == PGRERROR_TIMEOUT) {
            // ignore timeout errors, camera may not be responding to
            // register reads during power-up
        } else if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
        }

        retries--;
    } while ((regVal & k_powerVal) == 0 && retries > 0);

    // Check for timeout errors after retrying
    if (error == PGRERROR_TIMEOUT) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

int PGGigECam::setupTrigger() {

    // Get current trigger settings
    Error error = camera.GetTriggerModeInfo(&trigger_mode_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (trigger_mode_info.present != true) {
        std::cout << "Camera does not support external trigger. Exiting...\n";
        exit(EXIT_FAILURE);
    }

    TriggerMode triggerMode;
    error = camera.GetTriggerMode(&triggerMode);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    triggerMode.onOff = use_trigger;
    triggerMode.polarity = trigger_polarity;
    triggerMode.mode = trigger_mode;
    triggerMode.parameter = 0;
    triggerMode.source = trigger_source_pin;

    error = camera.SetTriggerMode(&triggerMode);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    // TODO: This hangs...
    //Poll to ensure camera is ready
//    if (use_trigger) { // If false, camera will free run
//        bool retVal = pollForTriggerReady();
//        if (!retVal) {
//            std::cout << "\n";
//            std::cout << "Error polling for trigger ready. Exiting...\n";
//            exit(EXIT_FAILURE);
//        }
//    }

    // Camera is ready, start capturing images
    error = camera.StartCapture();
    if (error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
        std::cout << "Bandwidth exceeded. Cannot start camera.\n";
        exit(EXIT_FAILURE);
    } else if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    aquisition_started = true;
    if (use_trigger && use_software_trigger) {
        std::cout << "Trigger the camera by pressing Enter\n";
    } else if(use_trigger) {
        std::cout << "Trigger the camera by sending a trigger pulse to GPIO_" << triggerMode.source << "\n";
    }else {
        std::cout << "Camera by started in free running mode.\n";
    }

    return 0;
}

void PGGigECam::grabImage() {

    // Get the image
    if (!aquisition_started) {
        std::cout << "Cannot grab image because acquisition has not been started.\n";
        exit(EXIT_FAILURE);
    }

    Error error = camera.RetrieveBuffer(&raw_image);
    if (error == PGRERROR_IMAGE_CONSISTENCY_ERROR) {
        std::cout << "WARNING: torn image detected.\n";

        // TODO: implement onboard buffer and perform retry a RetrieveBuffer
        // A single time if a torn image is detected.
    } else if (error != PGRERROR_OK) {
        printError(error);
        std::cout << "WARNING: capture error.\n";
    }
}

cv::Mat PGGigECam::imageToMat() {

    // convert to rgb
    raw_image.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgb_image);

    // convert to OpenCV cv::Mat
    unsigned int rowBytes = (double) rgb_image.GetReceivedDataSize() / (double) rgb_image.GetRows();
    return cv::Mat(rgb_image.GetRows(), rgb_image.GetCols(), CV_8UC3, rgb_image.GetData(), rowBytes);

}

void PGGigECam::grabMat() {

    grabImage();
    cvmat_image = imageToMat();
}

void PGGigECam::serveMat() {

    // Write frame to shared memory and notify all client processes
    // that a new frame is available. Do not block, though.
    frame_sink.pushMat(cvmat_image);
}

// PRIVATE

int PGGigECam::findNumCameras(void) {

    Error error;
    BusManager busMgr;

    error = busMgr.GetNumOfCameras(&num_cameras);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

void PGGigECam::printError(Error error) {
    error.PrintErrorTrace();
    if (!camera.IsConnected()) {
        std::cerr << "Camera must be connected before getting its info.\n";
    }
}

bool PGGigECam::pollForTriggerReady() {

    const unsigned int k_softwareTrigger = 0x62C;
    Error error;
    unsigned int regVal = 0;

    do {
        error = camera.ReadRegister(k_softwareTrigger, &regVal);
        if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
        }

    } while ((regVal >> 31) != 0);

    return true;
}

int PGGigECam::printBusInfo(void) {

    std::cout << "\n";
    std::cout << "*** BUS INFORMATION ***\n";
    std::cout << "No. cameras detected on bus: " << num_cameras << "\n";
    return 0;
}

int PGGigECam::printCameraInfo(void) {

    Error error = camera.GetCameraInfo(&camera_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    std::ostringstream macAddress;
    macAddress << std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[0] << ":" <<
            std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[1] << ":" <<
            std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[2] << ":" <<
            std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[3] << ":" <<
            std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[4] << ":" <<
            std::hex << std::setw(2) << std::setfill('0') << (unsigned int) camera_info.macAddress.octets[5];


    std::ostringstream ipAddress;
    ipAddress << (unsigned int) camera_info.ipAddress.octets[0] << "." <<
            (unsigned int) camera_info.ipAddress.octets[1] << "." <<
            (unsigned int) camera_info.ipAddress.octets[2] << "." <<
            (unsigned int) camera_info.ipAddress.octets[3];

    std::ostringstream subnetMask;
    subnetMask << (unsigned int) camera_info.subnetMask.octets[0] << "." <<
            (unsigned int) camera_info.subnetMask.octets[1] << "." <<
            (unsigned int) camera_info.subnetMask.octets[2] << "." <<
            (unsigned int) camera_info.subnetMask.octets[3];

    std::ostringstream defaultGateway;
    defaultGateway << (unsigned int) camera_info.defaultGateway.octets[0] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[1] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[2] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[3];

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
    std::cout << "GigE version :" << camera_info.gigEMajorVersion << "." << camera_info.gigEMinorVersion << "\n";
    std::cout << "User defined name :" << camera_info.userDefinedName << "\n";
    std::cout << "XML URL 1: " << camera_info.xmlURL1 << "\n";
    std::cout << "XML URL 2: " << camera_info.xmlURL2 << "\n";
    std::cout << "MAC address: " << macAddress.str() << "\n";
    std::cout << "IP address: " << ipAddress.str() << "\n";
    std::cout << "Subnet mask: " << subnetMask.str() << "\n";
    std::cout << "Default gateway: " << defaultGateway.str() << "\n\n";


    return 0;
}

void PGGigECam::printStreamChannelInfo(GigEStreamChannel *pStreamChannel) {
    //char ipAddress[32];
    std::ostringstream ipAddress;
    ipAddress << (unsigned int) pStreamChannel->destinationIpAddress.octets[0] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[1] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[2] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[3];

    std::cout << "Network interface: " << pStreamChannel->networkInterfaceIndex << "\n";
    std::cout << "Host Port: " << pStreamChannel->hostPort << "\n";
    std::cout << "Do not fragment bit: " << (pStreamChannel->doNotFragment ? "Enabled" : "Disabled") << "\n";
    std::cout << "Packet size: " << pStreamChannel->packetSize << "\n";
    std::cout << "Inter packet delay: " << pStreamChannel->interPacketDelay << "\n";
    std::cout << "Destination IP address: " << ipAddress.str() << "\n";
    std::cout << "Source port (on camera): " << pStreamChannel->sourcePort << "\n\n";

}

void PGGigECam::fireSoftwareTrigger() {
    // Check that the trigger is ready

    if (use_trigger && use_software_trigger) {

        pollForTriggerReady();

        std::cout << "Press the Enter key to initiate a software trigger\n";
        std::cin.ignore();

        // Fire software trigger
        const unsigned int k_softwareTrigger = 0x62C;
        const unsigned int k_fireVal = 0x80000000;
        Error error;

        error = camera.WriteRegister(k_softwareTrigger, k_fireVal);
        if (error != PGRERROR_OK) {
            printError(error);
            std::cout << "Error firing software trigger\n";
        }

    } else {

        std::cout << "Cannot firing software trigger because software trigger has not been configured.\n";
    }
}