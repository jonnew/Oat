#ifndef CameraControl_H
#define CameraControl_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "FlyCapture2.h"

#include "../../lib/shmem/MatServer.h"

class CameraControl : public MatServer {
   
public:
    CameraControl(std::string name);
    
    // Use a configuration file to specify parameters
    void configure(std::string config_file, std::string key);

    // For establishing connection
    int setCameraIndex(unsigned int requested_idx);
    int connectToCamera(void);

    // Once connected
    int setupStreamChannels(void);
    int setupShutter(float shutter_ms);
    int setupShutter(bool is_auto);
    int setupGain(float gain_dB);
    int setupGain(bool is_auto);
    int setupExposure(float exposure_EV);
    int setupExposure(bool is_auto);
    int setupWhiteBalance(int white_bal_red, int white_bal_blue);
    int setupWhiteBalance(bool is_on);
    int setupImageFormat(void);
    int setupDefaultImageFormat(void);
    //TODO: int setupImageFormat(int xOffset, int yOffset, int height, int width, PixelFormat format);
    //int setupImageBinning(int xBinFactor, int yBinFactor);
    int setupTrigger(int source, int polarity);

    // Physical camera control
    int turnCameraOn(void);
    //TODO: int turnCameraOff(void);
    void grabImage(void);
    void grabMat(cv::Mat& image);
    cv::Mat imageToMat(void);

    // IPC
    void createSharedBlock(void);
    void serveMat(void);

    // Accessors
    cv::Size get_frame_size(void) {
        return frame_size;
    }

private:

    // Camera name
    std::string camera_name;
    
    // Size and offset of the image to aquire
    cv::Size frame_size, frame_offset;
    
    // If we are serving the data, this is the currently captured image and
    // data size is the number of bytes in its data block
    cv::Mat mat;
    int data_size;
    
    bool aquisition_started;
    unsigned int num_cameras, index;
    float gain_dB, shutter_ms, exposure_EV;
    int white_bal_red, white_bal_blue;
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
