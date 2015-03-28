/* Camera configuration for rat-vision*/

#include "CameraControl.h"

#include <stdlib.h> 
#include <unistd.h> // Don't know how to include only if LINUX with cmake
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stdafx.h"
#include "FlyCapture2.h"

using namespace FlyCapture2;

CameraControl::CameraControl(void) {

    // Initialize the frame size
    // TODO: Hardcoded for the max square image on blackfly 09C
    frame_size = cv::Size(728, 728);

    // Start with 0 cameras on bus
    num_cameras = 0;
    index = 0;
    aquisition_started = false;
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

int CameraControl::setupShutter(float shutter_ms) {

    std::cout << "Setting up shutter..." << std::endl; 
    
    Property prop;
    prop.type = SHUTTER;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.autoManualMode = false;
    prop.absControl = true;
    prop.absValue = shutter_ms;
    
    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    std::cout << "Shutter time set to " << std::fixed << std::setprecision(2) << shutter_ms << " ms." << std::endl;
    
    return 0;
}

int CameraControl::setupGain(float gain_dB) {

    std::cout << "Setting camera gain..." << std::endl; 
    
    Property prop;
    prop.type = GAIN;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.autoManualMode = false;
    prop.absControl = true;
    prop.absValue = gain_dB;
    
    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    std::cout << "Gain set to " << std::fixed << std::setprecision(2) << gain_dB << " dB." << std::endl;
    
    return 0;
}

int CameraControl::setupExposure(float exposure_EV) {

    std::cout << "Setting up exposure..." << std::endl; 
    
    Property prop;
    prop.type = SHUTTER;
    Error error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.autoManualMode = true;
    
    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.type = GAIN;
    error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.autoManualMode = true;
    
    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.type = AUTO_EXPOSURE;
    error = camera.GetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    prop.onOff = true;
    prop.autoManualMode = false;
    prop.absControl = true;
    prop.absValue = exposure_EV;
    
    error = camera.SetProperty(&prop);
    if (error != PGRERROR_OK) {
            printError(error);
            exit(EXIT_FAILURE);
    }
    
    std::cout << "Exposure set to " << std::fixed << std::setprecision(2) << exposure_EV << " EV." << std::endl;
    
    return 0;
}

/**
 * Default image setup. Image uses all available pixels and BRG pixel format. 
 * 
 * @return 0 if successful.
 */
int CameraControl::setupImageFormat() {

    std::cout << "Querying GigE image setting information..." << std::endl;

    Error error = camera.GetGigEImageSettingsInfo(&image_settings_info);
    if (error != PGRERROR_OK) {
        printError(error);
        return -1;
    }

    GigEImageSettings imageSettings;
    imageSettings.offsetX = 0;
    imageSettings.offsetY = 0;
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

    // TODO: Triggering the camera externally using provided source and polarity
    //trig_mode.source = source;
    //trig_mode.polarity = polarity;

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

void CameraControl::grabImage(cv::Mat& image) {

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
    }
    else if (error != PGRERROR_OK) {
        printError(error);
        std::cout << "WARNING: capture error." << std::endl;
    }

    std::cout << "Grabbed image " << std::endl;

    // convert to rgb
    raw_image.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgb_image);

    // convert to OpenCV cv::Mat
    unsigned int rowBytes = (double) rgb_image.GetReceivedDataSize() / (double) rgb_image.GetRows();
    image = cv::Mat(rgb_image.GetRows(), rgb_image.GetCols(), CV_8UC3, rgb_image.GetData(), rowBytes);
    //std::cout << " Image = " << std::endl << " " << image << std::endl << std::endl;

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