/* Camera configuration for rat tracker use */
#ifndef CameraConfig_H
#define CameraConfig_H

#include <string.h>
// TODO: #defines for setting camera trigger modes

// Forward declaration of CameraInfo class
namespace FlyCapture2 {
    class CameraInfo;
}

class CameraConfig {

    public:
        CameraConfig(void);
        void print_camera_info(FlyCapture2::CameraInfo *p_cam_info);
        int set_gpio_trigger(void); //TODO: GPIO type
   		int set_ip(string ip); 
		
    private:

};

#endif
