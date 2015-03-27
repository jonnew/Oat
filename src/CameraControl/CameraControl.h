#ifndef CameraControl_H
#define CameraControl_H

#include "FlyCapture2.h"
#include <opencv2/core/mat.hpp>
#include "SimpleTrackerConfig.h"

class CameraControl {
    
public:
    CameraControl(void);

    // For establishing connection
    int setCameraIndex(unsigned int requested_idx);
    int connectToCamera(void);

    // Once connected
    int setupStreamChannels(void);
    //int setupShutterAndGain(int shutter_ms, float gain_db);
    int setupImageFormat(void);
    //TODO: int setupImageFormat(int xOffset, int yOffset, int height, int width, PixelFormat format);
    //int setupImageBinning(int xBinFactor, int yBinFactor);
    int setupTrigger(int source, int polarity);

    // Physical camera control
    int turnCameraOn(void);
    //TODO: int turnCameraOff(void);
    void grabImage(cv::Mat& image);

    // Accessors
    inline cv::Size get_frame_size(void) {
        return frame_size;
    }

private:

    // Size of the image to aquire
    cv::Size frame_size;
    
    // TODO: Size of the offset box in lower right hand corner of sensory array 
    // to move the ROI of frame_size around
    //cv::Size frame_offset; 

    bool aquisition_started;
    unsigned int num_cameras, index;
    FlyCapture2::GigECamera camera;

    // Camera and control state info
    FlyCapture2::CameraInfo camera_info;
    FlyCapture2::TriggerModeInfo trigger_mode_info;
    FlyCapture2::GigEImageSettingsInfo image_settings_info;

    // The current, unbuffered frame
    FlyCapture2::Image raw_image;
    FlyCapture2::Image rgb_image;

    int findNumCameras(void);
    void printError(FlyCapture2::Error error);
    bool pollForTriggerReady(void);
    int printCameraInfo(void);
    int printBusInfo(void);
    void printStreamChannelInfo(FlyCapture2::GigEStreamChannel *stream_channel);
};

#endif //CameraConfig_H
