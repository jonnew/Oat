#ifndef MATSERVETEST_H
#define MATSERVETEST_H

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "../../lib/shmem/SharedMat.h"
#include "../../lib/shmem/MatServer.h"

class MatServeTest : public MatServer {
   
public:
    MatServeTest(std::string server_name);

    int openVideo(const std::string fid);
    int serveMat(void);
   
private:
    
    int data_size;
    void* shared_mat_data_ptr;
    cv::VideoCapture cap;
    
    // Image data
    cv::Mat mat;

};

#endif //MATSERVETEST_H
