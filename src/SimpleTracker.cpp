
#include <iostream>
#include "CameraControl.h"
#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace std;

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

    // Start image acquisition
    //cv::Mat image, res;
    //vector<cv::Mat> spl;
    cv::VideoWriter outputVideo;
    cv::Mat image = cv::Mat(cc.get_frame_size(), CV_8UC3);
    //cv::Mat res = cv::Mat(S, CV_8UC3);
    //image = cv::Mat::zeros(S, CV_8UC3);
    outputVideo.open("temp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 27, cc.get_frame_size(), true);

    if (!outputVideo.isOpened()) {
        cout << "Could not open the output video for write: " << endl;
        return -1;
    }

    char key = '0';
    while(key != 'q') {

        // Get image
        cc.grab_image(image);
        
        
        
        
        //        if (image.empty()) {
        //            cout << "The acquired image is empty." << endl;
        //            continue;
        //        }
        //        outputVideo << image;
        //        
        //        cv::split(image, spl);                // process - extract only the correct channel
        //        for (int i =0; i < 3; ++i)
        //            if (i != 2)
        //                spl[i] = cv::Mat::zeros(S, spl[0].type());
        //        cv::merge(spl, res);


        // Dump the acquired image to the command line for inspection
        //cout << " Image = " << endl << " " << image << endl << endl;
        cv::imshow("image", image);
        key = cv::waitKey(30);
    }

    // Exit
    return 0;

}
