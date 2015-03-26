/* 
 * File:   HSVFilter.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 25, 2015, 5:11 PM
 */

#ifndef HSVFILTER_H
#define	HSVFILTER_H

#include <string>

using namespace std;



class HSVFilter {
public:
    HSVFilter();
    HSVFilter(const HSVFilter& orig);
    virtual ~HSVFilter();
    
    
    // HSV Filter settings track bars
    void createTrackbars(void);

private:
    
    void HSVFilter::on_trackbar(int, void*);
    
    int H_MIN = 0;
    int H_MAX = 256;
    int S_MIN = 0;
    int S_MAX = 256;
    int V_MIN = 0;
    int V_MAX = 256;
    const string trackbarWindowName = "Trackbars";

};

#endif	/* HSVFILTER_H */

