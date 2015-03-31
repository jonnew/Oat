/* 
 * File:   Tracker.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 29, 2015, 3:43 PM
 */

#include "Tracker.h"

#include <opencv2/highgui/highgui.hpp> 
#include <string>
#include <iostream>
#include <cassert>

Tracker::Tracker(std::string config_file) {

    orig_image.copyTo(proc_image);

    try {
        config = cpptoml::parse_file(config_file);

        std::cout << "Parsed the following configuration..." << std::endl << std::endl;
        std::cout << config << std::endl;
    } catch (const cpptoml::parse_exception& e) {
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

    //    hsv_filters.clear();
    //    detectors.clear();
    //    filtered_images.clear();
}

void Tracker::build() {

    // See if a camera configuration was provided
    if (config.contains("camera")) {

        auto camera_config = *config.get_table("camera");

        // Set the camera index
        if (camera_config.contains("index"))
            cc.index = (unsigned int) (*camera_config.get_as<int64_t>("index"));
        else
            cc.setCameraIndex(0);

        cc.connectToCamera();
        cc.turnCameraOn();
        cc.setupStreamChannels();

        // Set the exposure
        if (camera_config.contains("exposure")) {
            cc.exposure_EV = (float) (*camera_config.get_as<double>("exposure"));
            cc.setupExposure(false);
        } else {
            // Default to auto exposure
            cc.setupExposure(true);
        }

        // Set the shutter time
        if (camera_config.contains("shutter") && !camera_config.contains("exposure")) {
            cc.shutter_ms = (float) (*camera_config.get_as<double>("shutter"));
            cc.setupShutter(false);
        } else {
            // Default to auto shutter
            cc.setupShutter(true);
        }

        // Set the gain
        if (camera_config.contains("gain")&& !camera_config.contains("exposure")) {
            cc.gain_dB = (float) (*camera_config.get_as<double>("gain"));
            cc.setupGain(false);
        } else {
            // Default to auto gain
            cc.setupGain(true);
        }

        // Set white balance
        if (camera_config.contains("white_bal")) {

            auto wb = *camera_config.get_table("white_bal");

            cc.white_bal_red = (int) (*wb.get_as<int64_t>("red"));
            cc.white_bal_blue = (int) (*wb.get_as<int64_t>("blue"));
            cc.setupWhiteBalance(true);
        } else {

            // Default: turn white balance off
            cc.setupWhiteBalance(false);
        }

        // Set the ROI
        if (camera_config.contains("roi")) {

            auto roi = *camera_config.get_table("roi");

            cc.frame_offset.width = (int) (*roi.get_as<int64_t>("x_offset"));
            cc.frame_offset.height = (int) (*roi.get_as<int64_t>("y_offset"));
            cc.frame_size.width = (int) (*roi.get_as<int64_t>("width"));
            cc.frame_size.height = (int) (*roi.get_as<int64_t>("height"));
            cc.setupImageFormat();
        } else {
            cc.setupDefaultImageFormat();
        }

        cc.setupTrigger(0, 1); // TODO: Trigger options, free running w/o trigger

    }

    // See if tracker configuration was provided
    if (config.contains("tracker")) {

        auto tracker_config = *config.get_table("tracker");

        if (tracker_config.contains("background_subtract")) {

            auto bs = *tracker_config.get_table("background_subtract");

            background_subtract_on = (*bs.get_as<bool>("on"));
            subtractor.show = (*bs.get_as<bool>("show"));
        }

        if (tracker_config.contains("detector")) {

            auto dt_array = *tracker_config.get_table_array("detector");

            // Cycle through the array to make the detectors
            auto it = dt_array.get().begin();

            while (it != dt_array.get().end()) {

                if ((*it)->contains("name")) {
                    
                    std::string name = *(*it)->get_as<std::string>("name");
                    HSVDetector* hsv_detector = new HSVDetector(name);
                    hsv_detectors.push_back(*hsv_detector);
                } else {
                    HSVDetector hsv_detector("generic");
                    hsv_detectors.push_back(hsv_detector);
                }

                if ((*it)->contains("show")) {
                    hsv_detectors.back().show = (*(*it)->get_as<bool>("show"));
                } else {
                    hsv_detectors.back().show = false;
                }
                
                if ((*it)->contains("position")) {
                    hsv_detectors.back().position = (*(*it)->get_as<std::string>("show"));
                } else {
                    hsv_detectors.back().position = "unknown";
                }

                if ((*it)->contains("erode")) {
                    hsv_detectors.back().set_erode_size((int) (*(*it)->get_as<int64_t>("erode")));
                } 
                
                if ((*it)->contains("dilate")) {
                    hsv_detectors.back().set_dilate_size((int) (*(*it)->get_as<int64_t>("dilate")));
                } 
                
                if ((*it)->contains("h_thresholds")) {
                     auto t = *(*it)->get_table("h_thresholds");
                     
                     if  (t.contains("min")) {
                         hsv_detectors.back().h_min = (int) (*t.get_as<int64_t>("min"));
                     }
                     if  (t.contains("max")) {
                         hsv_detectors.back().h_max = (int) (*t.get_as<int64_t>("max"));
                     }
                } 
                
                if ((*it)->contains("s_thresholds")) {
                     auto t = *(*it)->get_table("s_thresholds");
                     
                     if  (t.contains("min")) {
                         hsv_detectors.back().s_min = (int) (*t.get_as<int64_t>("min"));
                     }
                     if  (t.contains("max")) {
                         hsv_detectors.back().s_max = (int) (*t.get_as<int64_t>("max"));
                     }
                } 
                
                if ((*it)->contains("v_thresholds")) {
                     auto t = *(*it)->get_table("v_thresholds");
                     
                     if  (t.contains("min")) {
                         hsv_detectors.back().v_min = (int) (*t.get_as<int64_t>("min"));
                     }
                     if  (t.contains("max")) {
                         hsv_detectors.back().v_max = (int) (*t.get_as<int64_t>("max"));
                     }
                } 
                
                if ((*it)->contains("hsv_tune")) {
                    if (*(*it)->get_as<bool>("hsv_tune")) 
                        hsv_detectors.back().createTrackbars();
                }
                
                cv::Mat filt_img;
                filtered_images.push_back(filt_img);
                ++it;
            }
        } else {
            std::cerr << "Tracker: A detector configuration must be provided in the form of array of tables." << std::endl;
            exit(EXIT_FAILURE);
        }

    } else {

        std::cerr << "Tracker: A tracker configuration must be provided." << std::endl;
        exit(EXIT_FAILURE);
    }

    // See if tracker configuration was provided
    if (config.contains("io")) {


    } else {

        std::cout << "Tracker: WARNING: no save or stream location provided. Data will not go anywhere..." << std::endl;
        exit(EXIT_FAILURE);

    }

}

/**
 * Run the tracker
 * 
 * COMMANDS
 * q = quit
 * p = pause
 * s = restart
 * b = set current image as background (if applicable)
 * r = enter image registration sequence
 * 
 * @return exit code
 */
int Tracker::run() {


    bool paused = false;
    char key = '0';
    while (key != 'q') {

        if (key == 'p' && !paused) {
            paused = true;
        }

        if (paused) {
            paused = !(key == 's');
        } else {

            // Get image
            cc.grabImage(orig_image);
            cv::imshow("Original image", orig_image);

            if (background_subtract_on) {
                subtractor.subtractBackground(orig_image, proc_image);
            }

            int i = 0;
            for (auto& h : hsv_detectors) {
                h.applyFilter(proc_image, filtered_images.at(i));
                ++i;
            }


        }

        if (key == 'b') {
            subtractor.setBackgroundImage(orig_image);
        }

        key = cv::waitKey(1);
    }

    // Exit
    return 0;
}
