//cartoonify Camera

//#include <iostream>  
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include "opencv2/opencv.hpp"
//
//using namespace std;
//using namespace cv;
//
//void cartoonifyImage(Mat cameraFrame, Mat displayedFrame) {
//	Mat gray;
//	cvtColor(cameraFrame, gray, CV_BGR2GRAY);
//	const int MEDIAN_BLUR_FILTER_SIZE = 5;
//	medianBlur(gray, gray, MEDIAN_BLUR_FILTER_SIZE);
//	Mat edges;
//	const int LAPLACIAN_FILTER_SIZE = 5;
//	Laplacian(gray, edges, CV_8U, LAPLACIAN_FILTER_SIZE);
//	
//	Mat mask;
//	const int EDGES_THRESHOLD = 80;
//	threshold(edges, mask, EDGES_THRESHOLD, 255, THRESH_BINARY_INV);
//	//Mat gray;
//	//cvtColor(cameraFrame, gray, CV_BGR2GRAY);
//	//const int MEDIAN_BLUR_FILTER_SIZE = 7;
//	//medianBlur(gray, gray, MEDIAN_BLUR_FILTER_SIZE);
//	//Mat edges, edges2;
//	//Scharr(gray, edges, CV_8U, 1, 0);
//	//Scharr(gray, edges2, CV_8U, 1, 0, -1);
//	//edges += edges2; // Combine the x & y edges together.
//	//const int EVIL_EDGE_THRESHOLD = 12;
//	//Mat mask;
//	//threshold(edges, mask, EVIL_EDGE_THRESHOLD, 255, THRESH_BINARY_INV);
//	//medianBlur(mask, mask, 3);
//
//	Size size = cameraFrame.size();
//	Size smallSize;
//	smallSize.width = size.width / 8;
//	smallSize.height = size.height / 8;
//	Mat smallImg = Mat(smallSize, CV_8UC3);
//	resize(cameraFrame, smallImg, smallSize, 0, 0, INTER_LINEAR);
//
//	Mat yuv = Mat(smallSize, CV_8UC3);
//	cvtColor(smallImg, yuv, CV_BGR2YCrCb);
//
//	Mat tmp = Mat(smallSize, CV_8UC3);
//	int repetitions = 7; // Repetitions for strong cartoon effect.
//	for (int i = 0; i<repetitions; i++) {
//		int ksize = 1; // Filter size. Has a large effect on speed.
//		double sigmaColor = 9; // Filter color strength.
//		double sigmaSpace = 7; // Spatial strength. Affects speed.
//		bilateralFilter(smallImg, tmp, ksize, sigmaColor, sigmaSpace);
//		bilateralFilter(tmp, smallImg, ksize, sigmaColor, sigmaSpace);
//	}
//	Mat bigImg = Mat(size, CV_8UC3);;
//	resize(smallImg, bigImg, size, 0, 0, INTER_LINEAR);
//	//bigImg.setTo(255);
//	displayedFrame.setTo(0);
//	bigImg.copyTo(displayedFrame,mask);
//	flip(displayedFrame, displayedFrame, 1);
//
//}
//
////
//int main(int argc, char* argv[]) {
//	int cameraNumber = 0;
//	if (argc > 1)
//		cameraNumber = atoi(argv[1]);
//	// Get access to the camera.
//	VideoCapture camera;
//	camera.open(cameraNumber);
//	if (!camera.isOpened()) {
//		std::cerr << "ERROR: Could not access the camera or video!" <<
//			std::endl;
//		exit(1);
//	}
//	// Try to set the camera resolution.
//	camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//	while (true) {
//		// Grab the next camera frame.
//		Mat cameraFrame;
//		camera >> cameraFrame;
//		if (cameraFrame.empty()) {
//			std::cerr << "ERROR: Couldn't grab a camera frame." <<
//				std::endl;
//			exit(1);
//		}
//		// Create a blank output image, that we will draw onto.
//		Mat displayedFrame(cameraFrame.size(), CV_8UC3);
//		// Run the cartoonifier filter on the camera frame.
//		cartoonifyImage(cameraFrame, displayedFrame);
//		// Display the processed image onto the screen.
//		
//		// Draw the color face onto a black background.
//		Size size = cameraFrame.size();
//		Mat faceOutline = Mat::zeros(size, CV_8UC3);
//		Scalar color = CV_RGB(255, 255, 0); // Yellow.
//		int thickness = 4;
//		// Use 70% of the screen height as the face height.
//		int sw = size.width;
//		int sh = size.height;
//		int faceH = sh / 2 * 70 / 100; // "faceH" is the radius of the ellipse.
//									   // Scale the width to be the same shape for any screen width.
//		int faceW = faceH * 72 / 100;
//		// Draw the face outline.
//		ellipse(faceOutline, Point(sw / 2, sh / 2), Size(faceW, faceH),
//			0, 0, 360, color, thickness, CV_AA);
//		// Draw anti-aliased text.
//		int fontFace = FONT_HERSHEY_COMPLEX;
//		float fontScale = 1.0f;
//		int fontThickness = 2;
//		char *szMsg = "Put your face here";
//		putText(faceOutline, szMsg, Point(sw * 23 / 100, sh * 10 / 100),
//			fontFace, fontScale, color, fontThickness, CV_AA);
//		faceOutline.copyTo(displayedFrame,faceOutline);
//		imshow("Cartoonifier", displayedFrame);
//		// IMPORTANT: Wait for at least 20 milliseconds,
//		// so that the image can be displayed on the screen!
//		// Also checks if a key was pressed in the GUI window.
//		// Note that it should be a "char" to support Linux.
//		char keypress = cv::waitKey(20); // Need this to see anything!
//		if (keypress == 27) { // Escape Key
//							  // Quit the program!
//			break;
//		}
//	}//end while
//}

//Histograms

//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace std;
//int main(int argc, char** argv) {
//	char i;
//	/*if (argc != 2) {
//		cout << "Computer Color Histogram\nUsage: " << argv[0] << " <imagename>" << endl;
//		char i;
//		cin >> i;
//		return -1;
//	}*/
//	cv::Mat src = cv::imread("ArborSnowboard-Banner-01.jpg", 1);
//	if (src.empty()) { cout << "Cannot load " << "ArborSnowboard-Banner-01.jpg" << endl; cin >> i; return -1; }
//	// Compute the HSV image, and decompose it into separate planes.
//	//
//	cv::Mat hsv;
//	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
//	float h_ranges[] = { 0, 180 }; // hue is [0, 180]
//	float s_ranges[] = { 0, 256 };
//	const float* ranges[] = { h_ranges, s_ranges };
//	int histSize[] = { 30, 32 }, ch[] = { 0, 1 };
//	cv::Mat hist;
//	// Compute the histogram
//	//
//	cv::calcHist(&hsv, 1, ch, cv::noArray(), hist, 2, histSize, ranges, true);
//	cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
//	int scale = 10;
//	cv::Mat hist_img(histSize[0] * scale, histSize[1] * scale, CV_8UC3);
//	// Draw our histogram.
//	//
//	for (int h = 0; h < histSize[0]; h++) {
//		for (int s = 0; s < histSize[1]; s++) {
//			float hval = hist.at<float>(h, s);
//			cv::rectangle(
//				hist_img,
//				cv::Rect(h*scale, s*scale, scale, scale),
//				cv::Scalar::all(hval),
//				-1
//			);
//		}
//	}
//	cv::imshow("image", src);
//	cv::imshow("H-S histogram", hist_img);
//	cv::waitKey(0);
//	
//	return 0;
//	
//}

//Template Matching

//
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <opencv2/core/core.hpp>  
//#include "opencv2/opencv.hpp"
//#include <iostream>
//using namespace std;
//using namespace cv;
//char i;
//bool use_mask;
//Mat img; Mat templ; Mat mask; Mat result;
//const char* image_window = "Source Image";
//const char* result_window = "Result window";
//int match_method;
//int max_Trackbar = 5;
//void MatchingMethod(int, void*);
//int main(int argc, char** argv)
//{
//	/*if (argc < 3)
//	{
//		cout << "Not enough parameters" << endl;
//		cout << "Usage:\n./MatchTemplate_Demo <image_name> <template_name> [<mask_name>]" << endl;
//		return -1;
//	}*/
//	templ = imread("ArborSnowboard-Banner-01 120x80.jpg", IMREAD_COLOR);
//	if (argc > 3) {
//		use_mask = true;
//		mask = imread(argv[3], IMREAD_COLOR);
//	}
//	if ( templ.empty() || (use_mask && mask.empty()))
//	{
//		cout << "Can't read one of the images" << endl;
//		return -1;
//	}
//
//
//	int cameraNumber = 1;
//	if (argc > 1)
//		cameraNumber = atoi(argv[1]);
//	// Get access to the camera.
//	VideoCapture camera;
//	camera.open(cameraNumber);
//	if (!camera.isOpened()) {
//		std::cerr << "ERROR: Could not access the camera or video!" <<
//			std::endl;
//		exit(1);
//	}
//	// Try to set the camera resolution.
//	camera.set(CV_CAP_PROP_FRAME_WIDTH, 960);
//	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 540);
//
//	while (true) {
//		// Grab the next camera frame.
//		
//		camera >> img;
//		if (img.empty()) {
//			std::cerr << "ERROR: Couldn't grab a camera frame." <<
//				std::endl;
//			
//			exit(1);
//		}
//		namedWindow(image_window, WINDOW_AUTOSIZE);
//		namedWindow(result_window, WINDOW_AUTOSIZE);
//		const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
//		createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);
//		MatchingMethod(0, 0);
//		char keypress = cv::waitKey(20); // Need this to see anything!
//		if (keypress == 27) { // Escape Key
//							  // Quit the program!
//			break;
//		}
//	}
//	return 0;
//}
//void MatchingMethod(int, void*)
//{
//	Mat img_display;
//
//	img.copyTo(img_display);
//	int result_cols = img.cols - templ.cols + 1;
//	int result_rows = img.rows - templ.rows + 1;
//	result.create(result_rows, result_cols, CV_32FC1);
//	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
//	
//		matchTemplate(img, templ, result, match_method);
//	
//	
//	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
//
//	double minVal; double maxVal; Point minLoc; Point maxLoc;
//
//	Point matchLoc;
//	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
//	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
//	{
//		matchLoc = minLoc;
//	}
//	else
//	{
//		matchLoc = maxLoc;
//	}
//	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//	flip(img_display, img_display, 1);
//	imshow(image_window, img_display);
//	imshow(result_window, result);
//	return;
//}

// Video Playing 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
using namespace std;
int g_slider_position = 0;
int g_run = 1, g_dontset = 0; //start out in single step mode
cv::VideoCapture g_cap;
void onTrackbarSlide(int pos, void *) {
	g_cap.set(cv::CAP_PROP_POS_FRAMES, pos);
	if (!g_dontset)
		g_run = 1;
	g_dontset = 0;
}
int main(int argc, char** argv) {
	cv::namedWindow("Example2_4", cv::WINDOW_AUTOSIZE);
	g_cap.open("Video.MOV");
	int frames = (int)g_cap.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	cout << "Video has " << frames << " frames of dimensions("
		<< tmpw << ", " << tmph << ")." << endl;
	cv::createTrackbar("Position", "Example2_4", &g_slider_position, frames,
		onTrackbarSlide);
	cv::Mat frame;
	for (;;) {
		if (g_run != 0) {
			g_cap >> frame; if (frame.empty()) break;
			int current_pos = (int)g_cap.get(cv::CAP_PROP_POS_FRAMES);
			g_dontset = 1;
			cv::setTrackbarPos("Position", "Example2_4", current_pos);
			cv::imshow("Example2_4", frame);
			g_run -= 1;
		}
		char c = (char)cv::waitKey(10);
		if (c == 's') // single step
		{
			g_run = 1; cout << "Single step, run = " << g_run << endl;
		}
		if (c == 'r') // run mode
		{
			g_run = -1; cout << "Run mode, run = " << g_run << endl;
		}
		if (c == 27)
			break;
	}
	return(0);
}

// Fast Template 
//
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

///**
//* Function to perform fast template matching with image pyramid
//*/
//void fastMatchTemplate(cv::Mat& srca,  // The reference image
//	cv::Mat& srcb,  // The template image
//	cv::Mat& dst,   // Template matching result
//	int maxlevel)   // Number of levels
//{
//	std::vector<cv::Mat> refs, tpls, results;
//
//	// Build Gaussian pyramid
//	cv::buildPyramid(srca, refs, maxlevel);
//	cv::buildPyramid(srcb, tpls, maxlevel);
//
//	cv::Mat ref, tpl, res;
//
//	// Process each level
//	for (int level = maxlevel; level >= 0; level--)
//	{
//		ref = refs[level];
//		tpl = tpls[level];
//		res = cv::Mat::zeros(ref.size() + cv::Size(1, 1) - tpl.size(), CV_32FC1);
//
//		if (level == maxlevel)
//		{
//			// On the smallest level, just perform regular template matching
//			cv::matchTemplate(ref, tpl, res, CV_TM_CCORR_NORMED);
//		}
//		else
//		{
//			// On the next layers, template matching is performed on pre-defined 
//			// ROI areas.  We define the ROI using the template matching result 
//			// from the previous layer.
//
//			cv::Mat mask;
//			cv::pyrUp(results.back(), mask);
//
//			cv::Mat mask8u;
//			mask.convertTo(mask8u, CV_8U);
//
//			// Find matches from previous layer
//			std::vector<std::vector<cv::Point> > contours;
//			cv::findContours(mask8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//			// Use the contours to define region of interest and 
//			// perform template matching on the areas
//			for (int i = 0; i < contours.size(); i++)
//			{
//				cv::Rect r = cv::boundingRect(contours[i]);
//				cv::matchTemplate(
//					ref(r + (tpl.size() - cv::Size(1, 1))),
//					tpl,
//					res(r),
//					CV_TM_CCORR_NORMED
//				);
//			}
//		}
//
//		// Only keep good matches
//		cv::threshold(res, res, 0.94, 1., CV_THRESH_TOZERO);
//		results.push_back(res);
//	}
//
//	res.copyTo(dst);
//}
//
//int main()
//{
//	cv::Mat ref = cv::imread("ArborSnowboard-Banner-01.jpg");
//	cv::Mat tpl = cv::imread("ArborSnowboard template.jpg");
//	if (ref.empty() || tpl.empty())
//		return -1;
//
//	cv::Mat ref_gray, tpl_gray;
//	cv::cvtColor(ref, ref_gray, CV_BGR2GRAY);
//	cv::cvtColor(tpl, tpl_gray, CV_BGR2GRAY);
//
//	cv::Mat dst;
//	fastMatchTemplate(ref_gray, tpl_gray, dst, 2);
//
//	while (true)
//	{
//		double minval, maxval;
//		cv::Point minloc, maxloc;
//		cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);
//
//		if (maxval >= 0.9)
//		{
//			cv::rectangle(
//				ref, maxloc,
//				cv::Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),
//				CV_RGB(0, 255, 0), 2
//			);
//			cv::floodFill(
//				dst, maxloc,
//				cv::Scalar(0), 0,
//				cv::Scalar(.1),
//				cv::Scalar(1.)
//			);
//		}
//		else
//			break;
//	}
//
//	cv::imshow("result", ref);
//	cv::waitKey();
//	return 0;
//}