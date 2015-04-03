#ifndef CameraControl_H
#define CameraControl_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/SMClient.h"

class Viewer : public SMClient<cv::Mat> {
   
public:
    Viewer(std::string server_name);
    //Viewer(const Viewer& orig);
    //virtual ~Viewer();

    void showImage(void);
    void showImage(std::string title);
   
private:
    
    // Image data
    cv::Mat image;
    
    // Should the image be displayed?
    //bool show = false;

};

#endif //CameraConfig_H
