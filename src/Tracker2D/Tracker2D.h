#ifndef Tracker2D_H
#define Tracker2D_H


enum 

class Tracker2D {
public:
    Tracker2D(void);

    // Image buffering class
    // TODO: image buffer class	
    // TODO: provide access to particular color streams within the buffer

    // Measure2D class
    // TODO: Each measure object should focus on finding a single 2D point within a monochromatic (but potentially extracted from color - e.g. the red channel) image stream. Measurements from each measurement object are combined in this tracker class. 

    // Filter2D class 
    // TODO: Kalman2D (extendes generic Filter2D)
    
    set_color_channel(int channel) { color_channel = channel};

private:
    int color_channel;
};
#endif //Tracker2D_H



