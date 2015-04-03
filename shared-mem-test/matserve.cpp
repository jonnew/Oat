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


    shared_memory_object::remove("SM");
    named_mutex::remove("mtx");
    named_condition::remove("cnd");

    cv::Mat temp = cv::imread("test.png", CV_LOAD_IMAGE_COLOR);
    const int data_size = temp.total() * temp.elemSize();

    // Reserve shared memory
    managed_shared_memory shm(open_or_create, "SM", data_size + sizeof (SharedImageHeader) + 1024);

    // Pointer to shared memory region for the SharedImageHeader
    //int *i = shm.find_or_construct<int>("integer")(0);
    auto shared_mat_header = shm.find_or_construct<SharedImageHeader>("SharedMat")();

    // Unnamed shared memory for the mat data
    const auto shared_mat_data_ptr = shm.allocate(data_size);

    // write the size, type and image version to the Shared Memory.
    shared_mat_header->size = temp.size();
    shared_mat_header->type = temp.type();
    //shared_mat_header->version = 0;

    // Write the handler to the unnamed shared region holding the data
    shared_mat_header->handle = shm.get_handle_from_address(shared_mat_data_ptr);

    // Sync mechanisms
    named_mutex nmtx{open_or_create, "mtx"};
    named_condition ncnd{open_or_create, "cnd"};
    scoped_lock<named_mutex> lock{nmtx}; // This starts with the creator of the scoped lock owning the mutex.

    // Loop
    int i = 0;
    while ('q' != cv::waitKey(1)) {

        std::cout << i << std::endl;

        if (i % 2 == 0) {
            cv::bitwise_not(temp, temp);
        }

        // Memcopy the mat data to the shared block
        memcpy(shared_mat_data_ptr, temp.data, data_size);
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

