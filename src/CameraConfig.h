/* Camera configuration for rat tracker use */
#ifndef CameraConfig_H
#define CameraConfig_H

#include <string.h>
// TODO: #defines for setting camera trigger modes

// Forward declaration of CameraInfo class
namespace FlyCapture2 {
    class CameraInfo;
}


typedef enum {
	GPIO0,
	GPIO1,
	GPIO2,
	GPIO3,
	GPIO4,
} GPIO;


class CameraConfig {

    public:
        CameraConfig(void);
		void get_camera_info(void);
        void print_camera_info(void);
        int set_gpio_trigger(GPIO gpio); //TODO: GPIO type
   		int set_ip(string ip); 
   		int verify_ext_trigger_support(string ip); 
		
    private:
		FlyCapture2::CameraInfo &cam_info;
		FlyCapture2::TriggerMode &trig_mode;
		void print_error(void);
};

#endif
