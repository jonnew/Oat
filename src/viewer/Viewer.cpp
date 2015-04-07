

#include "Viewer.h"

#include <string>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "cpptoml.h"

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatClient.cpp"

using namespace boost::interprocess;

Viewer::Viewer(std::string server_name) : MatClient(server_name) 
{ 
// TODO: Settings specify window location

}

void Viewer::showImage() {
    
    showImage(name);
}

void Viewer::showImage(const std::string title) {
    
    if (!shared_mat_created) {
        findSharedMat();
    }
    
    sharable_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex);

    cv::imshow(title, get_shared_mat());
    cv::waitKey(1);
    
    shared_mat_header->cond_var.notify_all();
    shared_mat_header->cond_var.wait(lock);
}