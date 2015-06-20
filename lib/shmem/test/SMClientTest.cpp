#include "../SMClient.h"

SCENARIO("SMClients free shmem if ref count is 0", "[free-shmem]" ) {

    GIVEN( "A single SMClient" ) {
	
		std::string sm_name = "test";
		int shared_value;

		SMClient<int> client(sm_name);
		
		WHEN ( "the client destructor called" ) {
			THEN ( "shmem deallocated" ) {

			}
		}
	}		

	GIVEN( "Two SMClients with commom shmem" ) {
	
		std::string sm_name = "test";
		int shared_value;

		SMClient<int> client1(sm_name);
		SMClient<int> client2(sm_name);
		
		WHEN ( "the client1 destructor called" ) {
			THEN ( "shmem reference count is decremented" ) {

			}
		}
		WHEN ( "the client2 destructor called" ) {
			THEN ( "shmem deallocated" ) {

			}
		}
	}		
}
