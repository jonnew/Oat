/* Camera configuration for rat tracker use */
#ifndef CameraConfig_H
#define CameraConfig_H

#include "FlyCapture2.h"
#include <opencv2/core/core.hpp>
#include "SimpleTrackerConfig.h"

using namespace std;
using namespace FlyCapture2;

// TODO: #defines for setting camera trigger modes (Rising/falling)
typedef enum {
    GPIO0,
    GPIO1,
    GPIO2,
    GPIO3,
    GPIO4,
} GPIO;

class CameraControl {
    
public:
    CameraControl(void);
    
    // For establishing connection
    int print_bus_info(void);
    int set_camera_index(unsigned int requested_idx);
    int connect_to_camera(void);
    
    // Once connected
    int turn_camera_on(void);
    int setup_trigger(int source, int polarity);
    cv::Mat grab_image(void);
    
    // int turn_camera_off(void);
    void get_camera_info(void);
    int print_camera_info(void);
    
    
    //int set_ip(string ip);
    //int verify_ext_trigger_support(string ip);

private:
    bool aquisition_started;
    unsigned int num_cameras, index;
    Camera camera;
    CameraInfo cam_info;
    TriggerMode trig_mode;
    TriggerModeInfo trig_mode_info;
    PGRGuid guid;
    BusManager busMgr;
    

    int find_num_cameras(void); 
    void print_error(Error error);
    bool poll_for_trigger_ready(void);
    
};

#endif
