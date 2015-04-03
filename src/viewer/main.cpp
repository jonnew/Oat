
#include "Viewer.h"

int main(int argc, char *argv[]) {


    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " SERVER-NAME" << std::endl;
        std::cout << "Viewer for cv::Mat data servers" << std::endl;
        return 1;
    }

    Viewer viewer(argv[1]);

    while (1) {
        viewer.showImage();
    }

    // Exit
    return 0;
}
