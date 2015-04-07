
#include "MatServeTest.h"

int main(int argc, char *argv[]) {


    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << "  SERVER-NAME TEST-AVI-FILE" << std::endl;
        std::cout << "cv::Mat data server test" << std::endl;
        return 1;
    }

    MatServeTest server(argv[1]);
    server.openVideo(argv[2]);

    
    while (1) {
        std::cout << "Sent frame." << std::endl;
        server.serveMat();
    }

    // Exit
    return 0;
}
