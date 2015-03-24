#include "CameraControl.h"
#include "FlyCapture2.h"

int main(int argc, char *argv[]) {
   
    //FlyCapture2::Camera cam();

    // Create a CameraConfig object
    CameraControl cc;
    
    // Configure the camera and print the resulting configuration data
    cc.print_bus_info();
    cc.set_camera_index(0);
    cc.connect_to_camera();
    cc.print_camera_info();
    
    cc.setup_trigger(0, 1);
    
    // Start image aqusition
    while(1) {
        
        cc.grab_image();
    }

    // Exit
    return 0;

}   
