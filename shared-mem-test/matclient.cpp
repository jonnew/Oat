#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace boost::interprocess;

typedef struct {
    cv::Size size;
    int type;
    //int version;
    boost::interprocess::managed_shared_memory::handle_t handle;
} SharedImageHeader;

int main(int argc, char *argv[]) {

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
    named_mutex nmtx{open_or_create, "mtx"};
    named_condition ncnd{open_or_create, "cnd"};
    scoped_lock<named_mutex> lock{nmtx}; // This starts with the first creator of the scoped lock owning the mutex.
    
    int i = 0;
    while ('q' != cv::waitKey(40)) {

        std::cout << i << std::endl;

        cv::imshow("Client Window", shared);
        
        ++i;

        // We are done the shared block so inform other processes they can access.
        ncnd.notify_all();
        ncnd.wait(lock); // release ownership

    }

    // Last notify all must be called to inform all slave processes that
    // lock has be perminantly released (since we won't renter to while loop)
    ncnd.notify_all();

    return 0;
}

