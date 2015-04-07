
#include "CameraControl.h"

int main(int argc, char *argv[]) {


    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << "SINK-NAME CAMERA-SETTINGS-FILE" << std::endl;
        std::cout << "Data server for point-grey GigE cameras." << std::endl;
        return 1;
    }

    CameraControl cc(argv[1]);
    cc.configure(argv[2]);
    
    std::cout << "GigECamera server named " + cc.get_name() + " has started." << std::endl;

    // TODO: exit signal
    while (1) {

        cc.serveMat();
    }

    // Exit
    return 0;
}
