#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/sync/named_sharable_mutex.hpp>
//#include <boost/interprocess/sync/named_condition_any.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition_any.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace boost::interprocess;

typedef struct {
    cv::Size size;
    int type;
    //int version;
    boost::interprocess::managed_shared_memory::handle_t handle;
    boost::interprocess::interprocess_sharable_mutex mutex;
    boost::interprocess::interprocess_condition_any cond_var;
} SharedImageHeader;

int main(int argc, char *argv[]) {
    
    // Client number
    std::string cli_name= argv[1];

    // Open shared memory created by the server
    managed_shared_memory shm(open_only, "SM");

    // Pointer to shared memory region for the SharedImageHeader
    const auto shared_mat_header = shm.find<SharedImageHeader>("SharedMat").first; 
    if (!shared_mat_header) { // Does not work...
        exit(EXIT_FAILURE);
    }

    const cv::Mat shared(
            shared_mat_header->size,
            shared_mat_header->type,
            shm.get_address_from_handle(shared_mat_header->handle));

    // Sync mechanisms
    //named_sharable_mutex nmtx{open_or_create, "mtx"};
    //named_condition_any ncnd{open_or_create, "cnd"};
    sharable_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex); // This starts with the first creator of the scoped lock owning the mutex.
    
    int i = 0;
    while ('q' != cv::waitKey(40)) {

        cv::imshow("Client Window" + cli_name, shared);
        
        std::cout << i << std::endl;
        ++i;

        // We are done with the shared block so inform other processes they can access.
        shared_mat_header->cond_var.notify_all();
        shared_mat_header->cond_var.wait(lock);
        //ncnd.notify_all();
        //ncnd.wait(lock); // release ownership

    }

    // Last notify all must be called to inform all other processes that
    // mutex has released
    shared_mat_header->cond_var.notify_all();

    return 0;
}

