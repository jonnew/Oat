/* Camera configuration for rat-vision*/

#include "Viewer.h"

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "cpptoml.h"

//#include "MatServer.h"
#include "../../lib/shmem/SharedMat.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/SMClient.cpp"

Viewer::Viewer(std::string server_name) : SMClient<sharedmat::SharedMat>(server_name) { }


void Viewer::showImage() {
    
    image = sharedmat::constructImage(shared_object);
    cv::imshow(name, image);
    cv::waitKey(1);
}

void Viewer::showImage(const std::string title) {
    
    image = sharedmat::constructImage(shared_object);
    cv::imshow(title, image);
    cv::waitKey(1);
}