#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost::interprocess

int main {

	// Managed shared memory object
	managed_shared_memory shm = managed_shared_memory(open_or_create, "SM", 1024);
	int *i = shm.find_or_construct<int>("integer")(0);

	// Sync mechanisms
	named_mutex nmtx{open_or_create, "mtx"};
	named_condition ncnd{open_or_create, "cnd"};
	scoped_lock<named_mutex> lock{nmtx};

	// NOTE: this process automatically takes ownership of the 
	// lock before the while loop. Another process will block
	// until the wait() command is called by the named_condition 
	// on the lock to release ownership
	
	// Loop
	while (*i < 10) {

		if ( *i % 2 == 0) {
			++(*i);

			// We are done with i, so notify other processes
			// they can access shared memory
			ncnd.notify_all();
			ncnd.wait(lock); // release ownership
		}
		else {
			std::cout << "Value of i: " + *i << std::endl;
			++(*i);
			ncmd.notify_all();
			ncnd.wait(lock); // release ownership
		}

		shared_memory_object::remove("SM");
		named_mutex::remove("mtx");
		named_condition::remove("cnd");
	}
}

