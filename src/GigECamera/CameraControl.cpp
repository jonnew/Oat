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

#include "CameraControl.h"

#include <string>
#include <stdlib.h> 
#include <unistd.h> // Don't know how to include only if LINUX with cmake
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/core/core.hpp>

#include "FlyCapture2.h"

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/shmem/SharedMat.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/MatServer.cpp"

#include "stdafx.h"

using namespace FlyCapture2;

CameraControl::CameraControl(std::string name) : frame_sink(name) {

    // Initialize the frame size
    frame_size = cv::Size(728, 728);
    frame_offset = cv::Size(0, 0);

    // Start with 0 cameras on bus
    num_cameras = 0;
    index = 0;
    shutter_ms = 0;
    gain_dB = 0;
    exposure_EV = 0;
    aquisition_started = false;
}

void CameraControl::configure(std::string config_file, std::string key) {

    cpptoml::table config;

    try {
        config = cpptoml::parse_file(config_file);

        //std::cout << "Parsed the following configuration..." << std::endl << std::endl;
        //std::cout << config << std::endl;
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(key)) {

            auto camera_config = *config.get_table(key);
            camera_name = key;
          
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

            setupTrigger(0, 1); // TODO: Trigger options, free running w/o trigger

        } else {
            std::cerr << "No camera configuration named \"" + key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

}

int CameraControl::setCameraIndex(unsigned int requestedIdx) {

    // Find the number of cameras on the bus
    findNumCameras();

    // Print bus information and find the number of cameras on the bus
    printBusInfo();

    if (num_cameras > 0) {
        index = requestedIdx;

    } else {
        std::cerr << "No cameras were detected, so setting an index is not possible." << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

int CameraControl::connectToCamera(void) {

    std::cout << "Connecting to camera: " << index << std::endl;

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

int CameraControl::setupStreamChannels() {

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

        std::cout << "Printing stream channel information for channel " << i << std::endl;
        printStreamChannelInfo(&streamChannel);
    }

    return 0;
}

int CameraControl::setupShutter(bool is_auto) {

    std::cout << "Setting up shutter..." << std::endl;

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
        std::cout << "Shutter set to auto." << std::endl;
    } else {
        std::cout << "Shutter time set to " << std::fixed << std::setprecision(2) << shutter_ms << " ms." << std::endl;
    }

    return 0;
}

int CameraControl::setupShutter(float shutter_ms_in) {

    shutter_ms = shutter_ms_in;
    setupShutter(false);
}

int CameraControl::setupGain(bool is_auto) {

    std::cout << "Setting camera gain..." << std::endl;

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
        std::cout << "Gain set to auto." << std::endl;
    } else {
        std::cout << "Gain set to " << std::fixed << std::setprecision(2) << gain_dB << " dB." << std::endl;
    }

    return 0;
}

int CameraControl::setupGain(float gain_dB_in) {

    gain_dB = gain_dB_in;
    setupGain(false);

}

int CameraControl::setupExposure(bool is_auto) {

    std::cout << "Setting up exposure..." << std::endl;

    setupShutter(true);
    setupGain(true);
    //    Property prop;
    //    prop.type = SHUTTER;
    //    Error error = camera.GetProperty(&prop);
    //    if (error != PGRERROR_OK) {
    //            printError(error);
    //            exit(EXIT_FAILURE);
    //    }
    //    
    //    prop.autoManualMode = true;
    //    
    //    error = camera.SetProperty(&prop);
    //    if (error != PGRERROR_OK) {
    //            printError(error);
    //            exit(EXIT_FAILURE);
    //    }
    //    
    //    prop.type = GAIN;
    //    error = camera.GetProperty(&prop);
    //    if (error != PGRERROR_OK) {
    //            printError(error);
    //            exit(EXIT_FAILURE);
    //    }
    //    
    //    prop.autoManualMode = true;
    //    
    //    error = camera.SetProperty(&prop);
    //    if (error != PGRERROR_OK) {
    //            printError(error);
    //            exit(EXIT_FAILURE);
    //    }

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
        std::cout << "Exposure set to auto." << std::endl;
    } else {
        std::cout << "Exposure set to " << std::fixed << std::setprecision(2) << exposure_EV << " EV." << std::endl;
    }

    return 0;
}

int CameraControl::setupExposure(float exposure_EV_in) {

    exposure_EV = exposure_EV_in;
    setupExposure(false);
}

int CameraControl::setupWhiteBalance(bool is_on) {

    std::cout << "Setting camera white balance..." << std::endl;

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
        std::cout << "White balance set to: " << std::endl;
        std::cout << "\tRed: " << std::fixed << std::setprecision(2) << white_bal_red << std::endl;
        std::cout << "\tBlue: " << std::fixed << std::setprecision(2) << white_bal_blue << std::endl;
    } else {
        std::cout << "White balance turned off." << std::endl;
    }

    return 0;
}

int CameraControl::setupWhiteBalance(int white_bal_red_in, int white_bal_blue_in) {

    white_bal_red = white_bal_red_in;
    white_bal_blue = white_bal_blue_in;
    setupWhiteBalance(true);

}

/**
 * Default image setup. Image uses all available pixels and BRG pixel format. 
 * 
 * @return 0 if successful.
 */
int CameraControl::setupDefaultImageFormat() {

    std::cout << "Querying GigE image setting information..." << std::endl;

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

    std::cout << "Setting GigE image settings..." << std::endl;

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
int CameraControl::setupImageFormat() {

    std::cout << "Querying GigE image setting information..." << std::endl;

    Error error = camera.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (frame_offset.width > image_settings_info.maxWidth ||
            frame_offset.height > image_settings_info.maxHeight) {

        std::cerr << "ROI pixel offsets are larger than the CCD array. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((frame_offset.width + frame_size.width) > image_settings_info.maxWidth) {

        frame_size.width = image_settings_info.maxWidth - frame_offset.width;
        std::cout << "WARNING: Current X-axis ROI settings is off the CCD array" << std::endl;
        std::cout << "WARNING: Cropping the ROI to fit on the array: " << std::endl;
        std::cout << "\tFrame width: " + frame_size.width << std::endl;
        std::cout << "\tFrame offset: " + frame_offset.width << std::endl;
    }

    if ((frame_offset.height + frame_size.height) > image_settings_info.maxHeight) {

        frame_size.height = image_settings_info.maxHeight - frame_offset.height;
        std::cout << "WARNING: Current Y-axis ROI settings is off the CCD array" << std::endl;
        std::cout << "WARNING: Cropping the ROI to fit on the array: " << std::endl;
        std::cout << "\tFrame height: " + frame_size.height << std::endl;
        std::cout << "\tFrame offset: " + frame_offset.height << std::endl;
    }

    GigEImageSettings imageSettings;
    imageSettings.offsetX = frame_offset.width;
    imageSettings.offsetY = frame_offset.height;
    imageSettings.height = frame_size.height;
    imageSettings.width = frame_size.width;
    imageSettings.pixelFormat = PIXEL_FORMAT_RAW12;
    //imageSettings.pixelFormat = PIXEL_FORMAT_MONO8;

    std::cout << "Setting GigE image settings..." << std::endl;

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
int CameraControl::turnCameraOn() {

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

int CameraControl::setupTrigger(int source, int polarity) {

    // Get current trigger settings
    Error error = camera.GetTriggerModeInfo(&trigger_mode_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    if (trigger_mode_info.present != true) {
        std::cout << "Camera does not support external trigger. Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    TriggerMode triggerMode;
    error = camera.GetTriggerMode(&triggerMode);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    triggerMode.onOff = true;
    triggerMode.mode = 14; // Trigger Mode 14 (“Overlapped Exposure/Readout Mode”)
    triggerMode.parameter = 0;
    triggerMode.source = 0;

    error = camera.SetTriggerMode(&triggerMode);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    // Poll to ensure camera is ready
    bool retVal = pollForTriggerReady();
    if (!retVal) {
        std::cout << std::endl;
        std::cout << "Error polling for trigger ready. Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Camera is ready, start capturing images
    error = camera.StartCapture();
    if (error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
        std::cout << "Bandwidth exceeded. Cannot start camera." << std::endl;
        exit(EXIT_FAILURE);
    } else if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    aquisition_started = true;
    std::cout << "Trigger the camera by sending a trigger pulse to GPIO_" << triggerMode.source << std::endl;
    return 0;

}

void CameraControl::grabImage() {

    // Get the image
    if (!aquisition_started) {
        std::cout << "Cannot grab image because acquisition has not been started." << std::endl;
        exit(EXIT_FAILURE);
    }

    Error error = camera.RetrieveBuffer(&raw_image);
    if (error == PGRERROR_IMAGE_CONSISTENCY_ERROR) {
        std::cout << "WARNING: torn image detected." << std::endl;

        // TODO: implement onboard buffer and perform retry a RetrieveBuffer
        // A single time if a torn image is detected.
    } else if (error != PGRERROR_OK) {
        printError(error);
        std::cout << "WARNING: capture error." << std::endl;
    }

    std::cout << "Grabbed image " << std::endl;

}

cv::Mat CameraControl::imageToMat() {

    // convert to rgb
    raw_image.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgb_image);

    // convert to OpenCV cv::Mat
    unsigned int rowBytes = (double) rgb_image.GetReceivedDataSize() / (double) rgb_image.GetRows();
    return cv::Mat(rgb_image.GetRows(), rgb_image.GetCols(), CV_8UC3, rgb_image.GetData(), rowBytes);

}

void CameraControl::grabMat(cv::Mat& mat) {

    grabImage();
    mat = imageToMat();
}

void CameraControl::serveMat() {

    cv::Mat mat;
    grabMat(mat);
    frame_sink.set_shared_mat(mat); 
    frame_sink.notifyAllAndWait(); // Release exclusive lock on shared memory and wait
}

// PRIVATE

int CameraControl::findNumCameras(void) {

    Error error;
    BusManager busMgr;

    error = busMgr.GetNumOfCameras(&num_cameras);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

void CameraControl::printError(Error error) {
    error.PrintErrorTrace();
    if (!camera.IsConnected()) {
        std::cerr << "Camera must be connected before getting its info." << std::endl;
    }
}

bool CameraControl::pollForTriggerReady() {

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

int CameraControl::printBusInfo(void) {

    std::cout << std::endl;
    std::cout << "*** BUS INFORMATION ***" << std::endl;
    std::cout << "No. cameras detected on bus: " << num_cameras << std::endl;
    return 0;
}

int CameraControl::printCameraInfo(void) {

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

    std::cout << std::endl;
    std::cout << "*** GENERAL CAMERA INFORMATION ***" << std::endl;
    std::cout << "Serial number: " << camera_info.serialNumber << std::endl;
    std::cout << "Camera model: " << camera_info.modelName << std::endl;
    std::cout << "Camera vendor: " << camera_info.vendorName << std::endl;
    std::cout << "Sensor: " << camera_info.sensorInfo << std::endl;
    std::cout << "Resolution: " << camera_info.sensorResolution << std::endl;
    std::cout << "Firmware version: " << camera_info.firmwareVersion << std::endl;
    std::cout << "Firmware build time: " << camera_info.firmwareBuildTime << std::endl << std::endl;

    std::cout << "*** CAMERA INTERFACE INFORMATION ***" << std::endl;
    std::cout << "GigE version :" << camera_info.gigEMajorVersion << "." << camera_info.gigEMinorVersion << std::endl;
    std::cout << "User defined name :" << camera_info.userDefinedName << std::endl;
    std::cout << "XML URL 1: " << camera_info.xmlURL1 << std::endl;
    std::cout << "XML URL 2: " << camera_info.xmlURL2 << std::endl;
    std::cout << "MAC address: " << macAddress.str() << std::endl;
    std::cout << "IP address: " << ipAddress.str() << std::endl;
    std::cout << "Subnet mask: " << subnetMask.str() << std::endl;
    std::cout << "Default gateway: " << defaultGateway.str() << std::endl << std::endl;


    return 0;
}

void CameraControl::printStreamChannelInfo(GigEStreamChannel *pStreamChannel) {
    //char ipAddress[32];
    std::ostringstream ipAddress;
    ipAddress << (unsigned int) pStreamChannel->destinationIpAddress.octets[0] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[1] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[2] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[3];

    std::cout << "Network interface: " << pStreamChannel->networkInterfaceIndex << std::endl;
    std::cout << "Host Port: " << pStreamChannel->hostPort << std::endl;
    std::cout << "Do not fragment bit: " << (pStreamChannel->doNotFragment ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Packet size: " << pStreamChannel->packetSize << std::endl;
    std::cout << "Inter packet delay: " << pStreamChannel->interPacketDelay << std::endl;
    std::cout << "Destination IP address: " << ipAddress.str() << std::endl;
    std::cout << "Source port (on camera): " << pStreamChannel->sourcePort << std::endl << std::endl;

}