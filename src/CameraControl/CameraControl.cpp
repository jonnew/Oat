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


using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;
using std::hex;
using std::setw;
using std::setfill;
using cv::Mat;
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
        cerr << "No cameras were detected, so setting an index is not possible." << endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

int CameraControl::connectToCamera(void) {

    cout << "Connecting to camera: " << index << endl;

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

        cout << "Printing stream channel information for channel " << i << endl;
        printStreamChannelInfo(&streamChannel);
    }

    return 0;
}

//int CameraControl::setupShutterAndGain(int shutter_ms, float gain_db) {
//    
//    Property shutter = camera.GetProperty(&camera)
//    shutter.autoManualMode = false;
//    shutter.valueA = 1; // millisecond
//    camera.SetProperty(&shutter);
//    
//    Property gain = camera.GetProperty(GAIN);
//    gain.autoManualMode = false;
//    gain.absValue = gain_db;
//    camera.SetProperty(&gain);
//}

/**
 * Default image setup. Image uses all available pixels and BRG pixel format. 
 * 
 * @return 0 if successful.
 */
int CameraControl::setupImageFormat() {

    cout << "Querying GigE image setting information..." << endl;

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

    cout << "Setting GigE image settings..." << endl;

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
        cout << "Camera does not support external trigger. Exiting..." << endl;
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
        cout << endl;
        cout << "Error polling for trigger ready. Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

    // Camera is ready, start capturing images
    error = camera.StartCapture();
    if (error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED )
    {
        std::cout << "Bandwidth exceeded. Cannot start camera." << std::endl;     
        exit(EXIT_FAILURE);
    }
    else if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    aquisition_started = true;
    cout << "Trigger the camera by sending a trigger pulse to GPIO_" << triggerMode.source << endl;
    return 0;

}

void CameraControl::grabImage(cv::Mat& image) {

    // Get the image
    if (!aquisition_started) {
        cout << "Cannot grab image because acquisition has not been started." << endl;
        exit(EXIT_FAILURE);
    }
    
    Error error = camera.RetrieveBuffer(&raw_image);
    if (error == PGRERROR_IMAGE_CONSISTENCY_ERROR) {
        cout << "WARNING: torn image detected." << endl;
        
        // TODO: implement onboard buffer and perform retry a RetrieveBuffer
        // A single time if a torn image is detected.
    } 
    else if (error != PGRERROR_OK) {
        printError(error);
        cout << "WARNING: capture error." << endl;
    }

    cout << "Grabbed image " << endl;
    
    // convert to rgb
    raw_image.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgb_image);

    // convert to OpenCV Mat
    unsigned int rowBytes = (double) rgb_image.GetReceivedDataSize() / (double) rgb_image.GetRows();
    image = cv::Mat(rgb_image.GetRows(), rgb_image.GetCols(), CV_8UC3, rgb_image.GetData(), rowBytes);
    //cout << " Image = " << endl << " " << image << endl << endl;

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
        cerr << "Camera must be connected before getting its info." << endl;
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

    cout << endl;
    cout << "*** BUS INFORMATION ***" << endl;
    cout << "No. cameras detected on bus: " << num_cameras << endl;
    return 0;
}

int CameraControl::printCameraInfo(void) {

    Error error = camera.GetCameraInfo(&camera_info);
    if (error != PGRERROR_OK) {
        printError(error);
        exit(EXIT_FAILURE);
    }

    ostringstream macAddress;
    macAddress << hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[0] << ":" <<
            hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[1] << ":" <<
            hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[2] << ":" <<
            hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[3] << ":" <<
            hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[4] << ":" <<
            hex << setw(2) << setfill('0') << (unsigned int) camera_info.macAddress.octets[5];


    ostringstream ipAddress;
    ipAddress << (unsigned int) camera_info.ipAddress.octets[0] << "." <<
            (unsigned int) camera_info.ipAddress.octets[1] << "." <<
            (unsigned int) camera_info.ipAddress.octets[2] << "." <<
            (unsigned int) camera_info.ipAddress.octets[3];

    ostringstream subnetMask;
    subnetMask << (unsigned int) camera_info.subnetMask.octets[0] << "." <<
            (unsigned int) camera_info.subnetMask.octets[1] << "." <<
            (unsigned int) camera_info.subnetMask.octets[2] << "." <<
            (unsigned int) camera_info.subnetMask.octets[3];

    ostringstream defaultGateway;
    defaultGateway << (unsigned int) camera_info.defaultGateway.octets[0] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[1] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[2] << "." <<
            (unsigned int) camera_info.defaultGateway.octets[3];

    cout << endl;
    cout << "*** GENERAL CAMERA INFORMATION ***" << endl;
    cout << "Serial number: " << camera_info.serialNumber << endl;
    cout << "Camera model: " << camera_info.modelName << endl;
    cout << "Camera vendor: " << camera_info.vendorName << endl;
    cout << "Sensor: " << camera_info.sensorInfo << endl;
    cout << "Resolution: " << camera_info.sensorResolution << endl;
    cout << "Firmware version: " << camera_info.firmwareVersion << endl;
    cout << "Firmware build time: " << camera_info.firmwareBuildTime << endl << endl;

    cout << "*** CAMERA INTERFACE INFORMATION ***" << endl;
    cout << "GigE version :" << camera_info.gigEMajorVersion << "." << camera_info.gigEMinorVersion << endl;
    cout << "User defined name :" << camera_info.userDefinedName << endl;
    cout << "XML URL 1: " << camera_info.xmlURL1 << endl;
    cout << "XML URL 2: " << camera_info.xmlURL2 << endl;
    cout << "MAC address: " << macAddress.str() << endl;
    cout << "IP address: " << ipAddress.str() << endl;
    cout << "Subnet mask: " << subnetMask.str() << endl;
    cout << "Default gateway: " << defaultGateway.str() << endl << endl;


    return 0;
}

void CameraControl::printStreamChannelInfo(GigEStreamChannel *pStreamChannel) {
    //char ipAddress[32];
    ostringstream ipAddress;
    ipAddress << (unsigned int) pStreamChannel->destinationIpAddress.octets[0] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[1] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[2] << "." <<
            (unsigned int) pStreamChannel->destinationIpAddress.octets[3];

    cout << "Network interface: " << pStreamChannel->networkInterfaceIndex << endl;
    cout << "Host Port: " << pStreamChannel->hostPort << endl;
    cout << "Do not fragment bit: " << (pStreamChannel->doNotFragment ? "Enabled" : "Disabled") << endl;
    cout << "Packet size: " << pStreamChannel->packetSize << endl;
    cout << "Inter packet delay: " << pStreamChannel->interPacketDelay << endl;
    cout << "Destination IP address: " << ipAddress.str() << endl;
    cout << "Source port (on camera): " << pStreamChannel->sourcePort << endl << endl;

}