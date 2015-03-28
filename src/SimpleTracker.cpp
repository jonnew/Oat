
#include <iostream>
#include "CameraControl.h"
#include "Tracker/HSVFilter.h"
#include "Tracker/BackgroundSubtractor.h"
#include "Tracker/Tracker.h"
#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <opencv2/imgproc.hpp>


int main(int argc, char *argv[]) {

    //FlyCapture2::Camera cam();

    // Create a CameraConfig object
    CameraControl cc;
    BackgroundSubtractor br_subtract;
    
    // TODO: Settings file? Whats the most easily parsable and human readable
    // filetype in C++
    
    // TODO: Settable at runtime or input arguments
    HSVFilter hsv_filter_blue("Blue", 106, 126, 69, 256, 180, 256);
    HSVFilter hsv_filter_orange("Orange", 0, 32, 32, 210, 164, 256);
    //HSVFilter hsv_filter_blue("Blue");
    //HSVFilter hsv_filter_orange("Orange");
    
    Tracker tracker_blue;
    Tracker tracker_orange;

    // Configure the camera and print the resulting configuration data
    cc.setCameraIndex(0);
    cc.connectToCamera();
    cc.turnCameraOn();
    cc.setupExposure(-0.5); //TODO: CLI input arg
    cc.setupStreamChannels();
    cc.setupImageFormat();
    cc.setupTrigger(0, 1);

    // TODO: recorder class
    
    cv::Mat image = cv::Mat(cc.get_frame_size(), CV_8UC3);
    cv::Mat filt_image_blue = image.clone();
    cv::Mat filt_image_orange = image.clone();

    // Start image acquisition
    char key = '0';
    int i = 0;
    while(key != 'q') {
        
        // Get image
        cc.grabImage(image);
        //cv::imshow("Original image", image);
        
        // TODO: Allow setting at arbitrary times
//        if (i == 0) {
//            br_subtract.setBackgroundImage(image);
//        }
        //br_subtract.subtrackBackground(image, image);
        //imshow("Background subtract", image);
        
        // Apply HSV filter
        hsv_filter_blue.applyFilter(image, filt_image_blue);
        hsv_filter_orange.applyFilter(image, filt_image_orange);
        cv::imshow("Blue threshold transform", filt_image_blue);
        cv::imshow("Orange threshold transform", filt_image_orange);
        
        // Find objects in the filtered image
        tracker_blue.findObjects(filt_image_blue);
        tracker_orange.findObjects(filt_image_orange);
        
        tracker_blue.decorateFeed(image, cv::Scalar(0, 0, 255));
        tracker_orange.decorateFeed(image, cv::Scalar(255, 0, 0));
        cv::imshow("Original image", image);
        
        // Display
        key = cv::waitKey(1);
        
        i++;
    }

    // Exit
    return 0;

}
