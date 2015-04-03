
#include "CameraControl.h"

int main(int argc, char *argv[]) {


    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << "CAMERA-SETTINGS-FILE" << std::endl;
        std::cout << "Data server for point-grey GigE cameras." << std::endl;
        return 1;
    }

    CameraControl cc("camera-server");
    cc.configure(argv[1]);

    while (1) {

        cc.serveMat();

    }

    // Exit
    return 0;
}
