/* 
 * File:   HSVFilter.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 25, 2015, 5:11 PM
 */

#include "HSVFilter.h"
#include <string>
#include <iostream>

HSVFilter::HSVFilter() {
}

HSVFilter::HSVFilter(const HSVFilter& orig) {
}

HSVFilter::~HSVFilter() {
}

void HSVFilter::createTrackbars() {
    //create window for trackbars

    cv::namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf(TrackbarName, "H_MIN", H_MIN);
    sprintf(TrackbarName, "H_MAX", H_MAX);
    sprintf(TrackbarName, "S_MIN", S_MIN);
    sprintf(TrackbarName, "S_MAX", S_MAX);
    sprintf(TrackbarName, "V_MIN", V_MIN);
    sprintf(TrackbarName, "V_MAX", V_MAX);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH), 
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->      
    cv::createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
    cv::createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
    cv::createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
    cv::createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
    cv::createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
    cv::createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);

}

void HSVFilter::on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed





}


