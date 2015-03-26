#ifndef CameraControl_H
#define CameraControl_H

#include "FlyCapture2.h"
#include <opencv2/core/core.hpp>
#include "SimpleTrackerConfig.h"

using namespace std;
using namespace FlyCapture2;

#define	FRAME_WIDTH = 0;
#define	FRAME_H = 256;

class CameraControl {
public:
    CameraControl(void);

    // For establishing connection
    int set_camera_index(unsigned int requestedIdx);
    int connect_to_camera(void);

    // Once connected
    int setup_stream_channels(void);
    int setup_image_format();
    //int setup_image_format(int xOffset, int yOffset, int height, int width, PixelFormat format);
    //int setup_image_binning(int xBinFactor, int yBinFactor);
    int setup_trigger(int source, int polarity);

    // Physical camera control
    int turn_camera_on(void);
    //void dummy_grab_image(void);
    void grab_image(cv::Mat& image);
    // int turn_camera_off(void);
    //void get_camera_info(void);

    inline cv::Size get_frame_size(void) {return frameSize;}
    


private:
    
    // Frame width/height (hard coded for now)
    // max square image size with blackfly 09C
    const cv::Size frameSize = cv::Size(728, 728);

    bool aquisitionStarted;
    unsigned int numCameras, index;
    GigECamera camera;

    // Camera and control state info
    CameraInfo cameraInfo;
    TriggerModeInfo triggerModeInfo;
    GigEImageSettingsInfo imageSettingsInfo;

    // The current, unbuffered frame
    Image rawImage;
    Image rgbImage;

    int find_num_cameras(void);
    void print_error(Error error);
    bool poll_for_trigger_ready(void);
    int print_camera_info(void);
    int print_bus_info(void);
    void print_stream_channel_info(GigEStreamChannel *pStreamChannel);
};

#endif //CameraConfig_H
