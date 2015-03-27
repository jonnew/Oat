
#include <iostream>
#include "CameraControl.h"
#include "Tracker/HSVFilter.h"
#include "Tracker/BackgroundSubtractor.h"
#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using cv::Mat;
using cv::imshow;

int main(int argc, char *argv[]) {

    //FlyCapture2::Camera cam();

    // Create a CameraConfig object
    CameraControl cc;
    BackgroundSubtractor br_subtract;
    
    // TODO: Settable at runtime or input arguements
    HSVFilter hsv_filter_blue(106, 126, 69, 256, 147, 256);
    HSVFilter hsv_filter_orange(0, 32, 32, 256, 88, 256);

    // Configure the camera and print the resulting configuration data
    cc.setCameraIndex(0);
    cc.connectToCamera();
    cc.turnCameraOn();
    cc.setupStreamChannels();
    //cc.setupShutterAndGain(1, 0.0); TODO: Need to set exposure!!
    cc.setupImageFormat();
    cc.setupTrigger(0, 1);

    // TODO: recorder class
    // Video recorder
    //cv::VideoWriter outputVideo;
    //outputVideo.open("temp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 27, cc.get_frame_size(), true);
//    if (!outputVideo.isOpened()) {
//        cout << "Could not open the output video for write: " << endl;
//        return -1;
//    }
    
    Mat image = cv::Mat(cc.get_frame_size(), CV_8UC3);
    Mat filt_image_blue = image.clone();
    Mat filt_image_orange = image.clone();

    // Start image acquisition
    char key = '0';
    int i = 0;
    while(key != 'q') {
        
        // Get image
        cc.grabImage(image);
        imshow("Original image", image);
        
        // TODO: Allow setting at arbitrary times
        if (i == 0) {
            br_subtract.setBackgroundImage(image);
        }

        br_subtract.subtrackBackground(image, image);
        //imshow("Background subtract", image);
        
        // HSV filter
        hsv_filter_blue.applyFilter(image, filt_image_blue);
        hsv_filter_orange.applyFilter(image, filt_image_orange);
        imshow("Blue threshold transform", filt_image_blue);
        imshow("Orange threshold transform", filt_image_orange);
        
        // Display
        key = cv::waitKey(1);
        
        i++;
    }

    // Exit
    return 0;

}
