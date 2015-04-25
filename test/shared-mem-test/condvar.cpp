#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost::interprocess;

int main() {

    // Managed shared memory object
    managed_shared_memory shm = managed_shared_memory(open_or_create, "SM", 1024);
    int *i = shm.find_or_construct<int>("integer")(0);

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
    while (*i <= 10) {

        if (*i % 2 == 0) {
            
            std::cout << *i << std::endl;
            ++(*i);

            // We are done with i, so notify other processes
            // they can access shared memory
            ncnd.notify_all();
            ncnd.wait(lock); // release ownership
            
        } else {
            std::cout << *i << std::endl;
            ++(*i);
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

