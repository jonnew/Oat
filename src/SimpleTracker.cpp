#include "CameraControl.h"
#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

int main(int argc, char *argv[]) {
   
    //FlyCapture2::Camera cam();

    // Create a CameraConfig object
    CameraControl cc;
    
    // Configure the camera and print the resulting configuration data
    cc.set_camera_index(0);
    cc.connect_to_camera();
    cc.turn_camera_on();
    cc.setup_stream_channels();
    cc.setup_image_format();
    cc.setup_trigger(0, 1);
    
    // Start image aquisition
    cv::Mat img;
    cv::VideoWriter outputVideo; 
    cv::Size S = cv::Size((int) cc.get_image_info().maxWidth, cc.get_image_info().maxHeight);
    outputVideo.open("temp.avi", -1, 27, S, true);
    while(1) {
        
        outputVideo << cc.grab_image();
    }

    // Exit
    return 0;

}   
