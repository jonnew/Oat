#ifndef Tracker_H
#define Tracker_H


class Tracker {
public:
    Tracker(void);

    // Image buffering class
    // TODO: image buffer class	

    // Measure2D class
    // TODO: Each measure object should focus on finding a single 2D point within a monochromatic (but potentially extracted from color - e.g. the red channel) image stream. Measurements from each measurement object are combined in this tracker class. 

    // Filter2D class 
    // TODO: List of color streams to track
    
    // TODO: Kalman2D (extendes generic Filter2D)
    
    set_color_channel(int channel) { color_channel = channel};

private:
    
    std::List<HSVFilter> hsv_filters;
    int color_channel;
};
#endif //Tracker_H



