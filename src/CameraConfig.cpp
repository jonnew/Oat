/* Camera configuration for rat-vision*/
#include <iostream>
#include "CameraConfig.h"
#include "FlyCapture2.h"

CameraConfig::CameraConfig(void) {

}

void CameraConfig::print_camera_info(FlyCapture2::CameraInfo *p_cam_info) {
    std::cout << std::endl;
	std::cout << "*** GENERAL CAMERA INFORMATION ***" << std::endl;
	std::cout << "Serial number -" << p_cam_info->serialNumber << std::endl;
    std::cout << "Camera model - " << p_cam_info->modelName << std::endl;
    std::cout << "Camera vendor - " << p_cam_info->vendorName << std::endl;
    std::cout << "Sensor - " << p_cam_info->sensorInfo << std::endl;
    std::cout << "Resolution - " << p_cam_info->sensorResolution << std::endl;
    std::cout << "Firmware version - " << p_cam_info->firmwareVersion << std::endl;
    std::cout << "Firmware build time - " << p_cam_info->firmwareBuildTime << std::endl << std::endl;

    // Here we need to extend this method to print things about
    // triggering and other hardware settings specific to the CameraConfig
    // Class
	
}

int CameraConfig::set_gpio_trigger(void) {

    return 0;
}

