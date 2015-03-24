/* Camera configuration for rat-vision*/

#include "stdafx.h"
#include "FlyCapture2.h"
#include "CameraControl.h"
#include <stdlib.h> 
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <unistd.h> // Don't know how to include only if LINUX with cmake
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

CameraControl::CameraControl(void) {

    //Camera camera;
    //CameraInfo cam_info;
    //TriggerMode trig_mode;
    //BusManager busMgr;

    // Start with 0 cameras on bus
    num_cameras = 0;
    index = 0;
    aquisition_started = false;
}

void CameraControl::print_error(Error error) {
    error.PrintErrorTrace();
    if (!camera.IsConnected()) {
        std::cerr << "Camera must be connected before getting its info." << std::endl;
    }
}

int CameraControl::print_bus_info(void) {
    find_num_cameras();

    std::cout << std::endl;
    std::cout << "*** BUS INFORMATION ***" << std::endl;
    std::cout << "Cameras detected on bus: " << num_cameras << std::endl;
    return 0;
}

int CameraControl::set_camera_index(unsigned int requested_idx) {

    find_num_cameras();

    if (num_cameras > 0) {
        index = requested_idx;

    } else {
        std::cerr << "No cameras were detected, so setting an index is not possible." << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

int CameraControl::connect_to_camera(void) {

    Error error = busMgr.GetCameraFromIndex(0, &guid);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    // Connect to a camera
    error = camera.Connect(&guid);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    // Power on the camera
    const unsigned int k_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    error = camera.WriteRegister(k_cameraPower, k_powerVal);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

int CameraControl::turn_camera_on() {

    // Power on the camera
    const unsigned int k_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    Error error = camera.WriteRegister(k_cameraPower, k_powerVal);
    if (error != PGRERROR_OK) {
        print_error(error);
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
            print_error(error);
            exit(EXIT_FAILURE);
        }

        retries--;
    } while ((regVal & k_powerVal) == 0 && retries > 0);

    // Check for timeout errors after retrying
    if (error == PGRERROR_TIMEOUT) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    error = camera.GetTriggerModeInfo(&trig_mode_info);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

int CameraControl::setup_trigger(int source, int polarity) {

    if (trig_mode_info.present != true) {
        cout << "Camera does not support external trigger! Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

    trig_mode.onOff = true;
    trig_mode.mode = 0;
    trig_mode.parameter = 0;

    // Triggering the camera externally using source 0.
    trig_mode.source = source;
    trig_mode.polarity = polarity;

    Error error = camera.SetTriggerMode(&trig_mode);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    // Poll to ensure camera is ready
    bool retVal = poll_for_trigger_ready();
    if (!retVal) {
        cout << endl;
        cout << "Error polling for trigger ready!" << endl;
        exit(EXIT_FAILURE);
    }

    // Camera is ready, start capturing images
    error = camera.StartCapture();
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    aquisition_started = true;
    cout << "Trigger the camera by sending a trigger pulse to GPIO..." << trig_mode.source << endl;
    return 0;

}

cv::Mat CameraControl::grab_image() {


    // Get the image
    Image rawImage;
    Error error = camera.RetrieveBuffer(&rawImage);
    if (error != PGRERROR_OK) {
        std::cout << "capture error." << std::endl;

    }

    // convert to RGB
    Image rgbImage;
    rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);

    // convert to OpenCV Mat
    unsigned int rowBytes = (double) rgbImage.GetReceivedDataSize() / (double) rgbImage.GetRows();
    cv::Mat image = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);

    cv::imshow("image", image);

    return image;

}

bool CameraControl::poll_for_trigger_ready() {

    const unsigned int k_softwareTrigger = 0x62C;
    Error error;
    unsigned int regVal = 0;

    do {
        error = camera.ReadRegister(k_softwareTrigger, &regVal);
        if (error != PGRERROR_OK) {
            print_error(error);
            exit(EXIT_FAILURE);
        }

    } while ((regVal >> 31) != 0);

    return true;
}

int CameraControl::print_camera_info(void) {//(CameraInfo *cam_info) {

    Error error = camera.GetCameraInfo(&cam_info);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    std::cout << std::endl;
    std::cout << "*** GENERAL CAMERA INFORMATION ***" << std::endl;
    std::cout << "Serial number: " << cam_info.serialNumber << std::endl;
    std::cout << "Camera model: " << cam_info.modelName << std::endl;
    std::cout << "Camera vendor: " << cam_info.vendorName << std::endl;
    std::cout << "Sensor: " << cam_info.sensorInfo << std::endl;
    std::cout << "Resolution: " << cam_info.sensorResolution << std::endl;
    std::cout << "Firmware version: " << cam_info.firmwareVersion << std::endl;
    std::cout << "Firmware build time: " << cam_info.firmwareBuildTime << std::endl << std::endl;

    std::cout << "*** CAMERA INTERFACE INFORMATION ***" << std::endl;
    std::cout << "Interface type: " << cam_info.interfaceType << std::endl;
    std::cout << "MAC address: " << string(reinterpret_cast<const char*> (cam_info.macAddress.octets)) << std::endl;
    if (cam_info.interfaceType == INTERFACE_GIGE) {
        std::cout << "IP address: " << string(reinterpret_cast<const char*> (cam_info.ipAddress.octets)) << std::endl;
        std::cout << "Subnet mask: " << string(reinterpret_cast<const char*> (cam_info.subnetMask.octets)) << std::endl << std::endl;
    }

    return 0;
    // Here we need to extend this method to print things about
    // triggering and other hardware settings specific to the CameraConfig
    // Class
    // }

}

int CameraControl::find_num_cameras(void) {

    Error error;



    error = busMgr.GetNumOfCameras(&num_cameras);
    if (error != PGRERROR_OK) {
        print_error(error);
        exit(EXIT_FAILURE);
    }

    return 0;
}

