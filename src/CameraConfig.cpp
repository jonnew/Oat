/* Camera configuration for rat-vision*/
#include <iostream>
#include "CameraConfig.h"
#include "FlyCapture2.h"

CameraConfig::CameraConfig(void) {

	cam_info = 
}

void CameraConfig::print_camera_info(void) {
	if (cam_info != null) { 

		std::cout << std::endl;
		std::cout << "*** GENERAL CAMERA INFORMATION ***" << std::endl;
		std::cout << "Serial number -" << cam_info.serialNumber << std::endl;
		std::cout << "Camera model - " << cam_info.modelName << std::endl;
		std::cout << "Camera vendor - " << cam_info.vendorName << std::endl;
		std::cout << "Sensor - " << cam_info.sensorInfo << std::endl;
		std::cout << "Resolution - " << cam_info.sensorResolution << std::endl;
		std::cout << "Firmware version - " << cam_info.firmwareVersion << std::endl;
		std::cout << "Firmware build time - " << cam_info.firmwareBuildTime << std::endl << std::endl;

		// Here we need to extend this method to print things about
		// triggering and other hardware settings specific to the CameraConfig
		// Class
		// }

}

int CameraConfig::set_gpio_trigger(void) {

	return 0;
}

