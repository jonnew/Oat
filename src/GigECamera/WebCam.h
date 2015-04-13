#ifndef WEBCAM_H
#define WEBCAM_H

#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp> // TODO: correct header...

class WebCam {

	public:
		WebCam(std::string name);
 		
		// Configure
		void configure(void); // Defualt options
		void configure(std::string config_file, std::string key);

		// IPC
		void grabMat(cv::Mat& image);
		void serveMat(void);

	private:
		std::string camera_name;
		bool aquisition_started;

		// The webcam object
		cv::VideoCapture camera;
		
		// Currently acquired frame
		cv::Mat current_frame;

		// Shared memory server
		MatServer frame_sink;

#endif //WEBCAM_H
