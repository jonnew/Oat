/* 
 * File:   Tracker.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 29, 2015, 3:43 PM
 */

#include "Tracker.h"

#include <iostream>
#include <cassert>

Tracker::Tracker(std::string config_file) {

    try {
        config = cpptoml::parse_file(config_file);

        std::cout << "Parsed the following configuration..." << std::endl << std::endl;
        std::cout << config << std::endl;
    }    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << std::endl;
    }

    try {
        build();
    } catch (const std::exception& e) {
         std::cerr << e.what() << std::endl;
    }

}

Tracker::Tracker(const Tracker& orig) {
}

Tracker::~Tracker() {
}

void Tracker::build() {

    // See if a camera configuration was provided
    if (config.contains("camera")) {

        auto camera_config = *config.get_table("camera");
        bool using_exposure = false;

        // Set the camera index
        if (camera_config.contains("index")) 
            cc.index = (unsigned int)(*camera_config.get_as<int64_t>("index"));
        else
            cc.setCameraIndex(0);
        
        // Set the exposure
        if (camera_config.contains("exposure")) {
            cc.exposure_EV = (float)(*camera_config.get_as<double>("exposure"));
            using_exposure = true;
        }
        
        // Set the shutter time
        if (camera_config.contains("shutter")) {
            cc.shutter_ms = (float)(*camera_config.get_as<double>("shutter"));
        }
        
        // Set the gain
        if (camera_config.contains("gain")) {
            cc.gain_dB = (float)(*camera_config.get_as<double>("gain"));
        }
        
        // Run camera configuration
        cc.connectToCamera();
        cc.turnCameraOn();
        if (using_exposure) {
            cc.setupExposure();
        } 
        else {
            cc.setupGain();
            cc.setupShutter();
        }
        cc.setupStreamChannels();
        cc.setupImageFormat(); // TODO: ROI
        cc.setupTrigger(0, 1); // TODO: Trigger options

    }
    
    // See if tracker configuration was provided
    if (config.contains("tracker")) {
        
    
    }
    else {
        
        cerr << "A tracker configuration must be provided." << endl;
        exit(EXIT_FAILURE);
        
    }
    
        // See if tracker configuration was provided
    if (config.contains("io")) {
        
    
    }
    else {
        
        cout << "WARNING: no save or stream location provided. Data will not go anywhere..." << endl;
        exit(EXIT_FAILURE);
        
    }

    //




}
