/* 
 * File:   BackgroundSubtractorCUDA.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on June 25, 2015, 2:45 PM
 */

#ifndef BACKGROUNDSUBTRACTORCUDA_H
#define	BACKGROUNDSUBTRACTORCUDA_H

#include "FrameFilter.h"
#include <opencv2/core/cuda.hpp>

class BackgroundSubtractorCUDA : public FrameFilter {
public:

    /**
     * A GPU-based background subtractor.
     * Subtract a frame image from a frame stream. The background frame is 
     * the first frame obtained from the SOURCE frame stream, or can be 
     * supplied via configuration file.
     * @param source_name raw frame source name
     * @param sink_name filtered frame sink name
     */
    BackgroundSubtractorCUDA(const std::string& source_name, const std::string& sink_name);

    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    cv::Mat filter(cv::Mat& frame);
    
    void configure(const std::string& config_file, const std::string& config_key);
    
private:

    // Is the background frame set?
    bool background_set = false;

    // The background frame
    cv::cuda::GpuMat result_frame;
    cv::cuda::GpuMat current_frame;
    cv::cuda::GpuMat background_frame;
    
    // Set the background frame
    //void setBackgroundImage(const cv::cuda::GpuMat&);
    void setBackgroundImage(const cv::Mat&);

};

#endif	/* BACKGROUNDSUBTRACTORCUDA_H */

