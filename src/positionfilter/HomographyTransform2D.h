/* 
 * File:   HomgraphicTransform2D.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on June 10, 2015, 11:26 AM
 */

#ifndef HOMGRAPHICTRANSFORM2D_H
#define	HOMGRAPHICTRANSFORM2D_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "PositionFilter.h"

class HomographyTransform2D : public PositionFilter {
public:
    HomographyTransform2D(const std::string& position_source_name, const std::string& position_sink_name);

    oat::Position2D filterPosition(oat::Position2D& position_in);

    /**
     * Configure homgraphy matrix using a configuration file.
     * @param config_file Path to the configuration file
     * @param config_key Configuration file key specifying the table containing
     * a nested array specifying the homography matrix
     */
    void configure(const std::string& config_file, const std::string& config_key);

private:

    bool homography_valid;
    cv::Matx33d homography;
    
    // Filtered position
    oat::Position2D filtered_position;

    // tuning window name
    std::string tuning_image_title;

};

#endif	/* HOMGRAPHICTRANSFORM2D_H */

