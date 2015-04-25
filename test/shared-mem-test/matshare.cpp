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

    if (argv[1]) {
        shared_memory_object::remove("SM");
        named_mutex::remove("mtx");
        named_condition::remove("cnd");
    }

    cv::Mat temp = cv::imread("test.png", CV_LOAD_IMAGE_COLOR);
    const int data_size = temp.total() * temp.elemSize();

    // Reserve shared memory
    managed_shared_memory shm = managed_shared_memory(open_or_create, "SM", 5*data_size + sizeof (SharedImageHeader) + 1024);

    // Pointer to shared memory region for the SharedImageHeader
    int *i = shm.find_or_construct<int>("integer")(0);
    auto shared_mat_header = shm.find_or_construct<SharedImageHeader>("SharedMat")();

    // Unnamed shared memory for the mat data

    
    const auto shared_mat_data_ptr = shm.allocate(data_size);
    const cv::Mat shared(
            shared_mat_header->size,
            shared_mat_header->type,
            shm.get_address_from_handle(shared_mat_header->handle));

    // Sync mechanisms
    named_mutex nmtx{open_or_create, "mtx"};
    named_condition ncnd{open_or_create, "cnd"};
    scoped_lock<named_mutex> lock{nmtx}; // This starts with the creator of the scoped lock owning the mutex.

    // NOTE: this process automatically takes ownership of the 
    // mutex before the while loop. Another process will block
    // until the wait() command is called by the named_condition 
    // on the lock to release ownership

    // NOTE: An exculsive mutex is needed for writing, and a non-exclusive
    // mutex is needed for reading

    // Loop
    while (*i <= 1000) {

        if (*i % 2 == 0) {

            std::cout << *i << std::endl;
            //*mat = temp; //.clone(); // Read the file

            if (*i % 4) {
                cv::bitwise_not(temp, temp);
            }

            // write the size, type and image version to the Shared Memory.
            shared_mat_header->size = temp.size();
            shared_mat_header->type = temp.type();
            //shared_mat_header->version = 0;

            // Write the handler to the unnamed shared region holding the data
            shared_mat_header->handle = shm.get_handle_from_address(shared_mat_data_ptr);

            // Memcopy the mat data to the shared block
            memcpy(shared_mat_data_ptr, temp.data, data_size);
            //*data = *(temp.data);
            //*datastart = *temp.datastart;
            //*dataend = *temp.dataend;
            //*datalimit = *temp.datalimit;

            ++(*i);

            // We are done with i, so notify other processes
            // they can access shared memory
            ncnd.notify_all();
            ncnd.wait(lock); // release ownership

        } else {
            std::cout << *i << std::endl;


            //*(mat->data) = *data;
            //*mat->datastart = *datastart;
            //*mat->dataend = *dataend;
            //*mat->datalimit = *datalimit;
            ++(*i);

            cv::imshow("Slave process", shared);
            cv::waitKey(1);
            ncnd.notify_all();
            ncnd.wait(lock); // release ownership
        }

    }

    // Last notify all must be called to inform all slave processes that
    // lock has be perminantly released (since we won't renter to while loop)
    ncnd.notify_all();
    shared_memory_object::remove("SM");
    named_mutex::remove("mtx");
    named_condition::remove("cnd");

    return 0;
}

