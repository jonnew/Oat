

#include "MatServeTest.h"

#include <string>
#include <opencv2/core/core.hpp>
//#include "cpptoml.h"

//#include "MatServer.h"
#include "../../lib/shmem/SharedMat.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/MatServer.cpp"

MatServeTest::MatServeTest(std::string server_name) : MatServer(server_name) { }

int MatServeTest::openVideo(const std::string fid) {

    cap.open(fid); // open the default video
    if (!cap.isOpened()) // check if we succeeded
        return -1;
}

int MatServeTest::serveMat() {

    cap >> mat; // get a new frame from video
    if (mat.empty()) {
        return 0;
        usleep(100000);
    }
    
    // Thread-safe set to shared mat object
    set_shared_mat(mat);
    
    return 1;

}