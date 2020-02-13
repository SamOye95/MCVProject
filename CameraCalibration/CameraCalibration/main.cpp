#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

#define RED		Scalar(255, 0, 0)
#define GREEN	Scalar(0, 255, 0)
#define BLUE	Scalar(0, 0, 255)
#define WHITE	Scalar(255, 255, 255)
#define ORIGIN	Point3f(0.f)

using namespace std;
using namespace cv;

const float squareDim = 0.023f; // meters
const Size boardDim = Size(6, 9);

int framePerSecond = 20;
int minSavedImages = 5; //Minimum number of boards need to be found before calibration

bool cameraCalibrated = false; // intial state of camera calibratio
							  
// create the realword positions of the board
void createKnowBoardPositions(Size boardSize, float edgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0 ; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * edgeLength, i * edgeLength, 0.0f));
		}
	}
}

// Find the intersections on the chessboard image.
void getCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		// Buffer to hold detected points
		vector<Point2f> pointBuf; 
		bool found = findChessboardCorners(*iter, boardDim, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		if (found) 
		{
			//Push back all detected points from point buffer
			allFoundCorners.push_back(pointBuf);
		}
		if (showResults)
		{
			//Draw the chessboard corners
			drawChessboardCorners(*iter, boardDim, pointBuf, found);
			imshow("Corner detection", *iter);
			waitKey();
		}
	}
}

// The function for the actual camera calibration and creation of the camera matrix. 
void cameraCalibration(vector<Mat> calImages, Size boardSize, float edgeLen, Mat& camMat, Mat& distCoef)
{
	vector<vector<Point2f>> imgSpacePoints;
	getCorners(calImages, imgSpacePoints, false);
	vector<vector<Point3f>> worldCornerPoints(1); 
	createKnowBoardPositions(boardSize, edgeLen, worldCornerPoints[0]);
	worldCornerPoints.resize(imgSpacePoints.size(), worldCornerPoints[0]);

	// Rotation vectors and translation vectors
	vector<Mat> rVectors, tVectors; 
	// Distortion coefficients of 8 elements
	distCoef = Mat::zeros(8, 1, CV_64F); 

	calibrateCamera(worldCornerPoints, imgSpacePoints, boardSize, camMat, distCoef, rVectors , tVectors);
}

// Save the camera calibration to a file
bool saveCamCalibration(string filename, Mat cameraMat, Mat distCoef)
{
	ofstream outStream(filename);
	if (outStream)
	{
		// Distortion coefficients
		uint16_t rows = cameraMat.rows;
		uint16_t cols = cameraMat.cols; 

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = cameraMat.at<double>(r, c);
				cout << value << "\n";
				outStream << value << endl; 
			}

		}
		// Distortion coefficients
		rows = distCoef.rows; 
		cols = distCoef.cols; 

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = distCoef.at<double>(r, c);
				cout << value << "\n";
				outStream << value << endl;
			}
		}

		outStream.close();
		cout << "Camera Calibration saved.\n";
		return true;

	}
	cout << "Camera Calibration not saved.\n";
	return false; 
}

// Load camera calibration matrix from a plain text file
bool loadCameraCalibration(string filename, Mat& cameraMat, Mat& distCoef) 
{
	ifstream inStream(filename);
	if (inStream) 
	{
		uint16_t rows;
		uint16_t columns;

		// Camera matrix
		inStream >> rows;
		inStream >> columns;

		cameraMat = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = 0.0f;
				inStream >> value;
				cameraMat.at<double>(r, c) = value;
				cout << cameraMat.at<double>(r, c) << "\n";
			}
		}

		//Distortion coefficients
		inStream >> rows;
		inStream >> columns;

		distCoef = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = 0.0f;
				inStream >> value;
				distCoef.at<double>(r, c) = value;
				cout << distCoef.at<double>(r, c) << "\n";
			}
		}
		inStream.close();
		return true;
	}
	return false;
}

// Draws an arrowed line projected to the world coordinate system using the camera intrinsic/extrinsics, in the given colour. 
void drawAxis(float x, float y, float z, Scalar color, Mat rVectors, Mat tVectors, Mat& cameraMat, Mat& distCoef, Mat& image)
{
	vector<Point3f> points;
	vector<Point2f> projectedPoints;
	
	//Fills input array with 2 points
	points.push_back(ORIGIN);
	points.push_back(Point3f(x, y, -z));

	// Projects points using projectPoints method
	projectPoints(points, rVectors, tVectors, cameraMat, distCoef, projectedPoints);
	
	// Draws corresponding line
	arrowedLine(image, projectedPoints[0], projectedPoints[1], color);
}

// Draws a cube with given side length and thickness and color, given the camera parameters.
void drawCube(float length, int thickness, Scalar color, Mat rVectors, Mat tVectors, Mat& cameraMat, Mat& distCoef, Mat& image) 
{
	vector<Point3f> points;
	vector<Point2f> projectedPoints;
	// Declare cube points from the origin
	points.push_back(Point3f(0.f, 0.f, 0.f)); // Point 0
	points.push_back(Point3f(length, 0.f, 0.f)); // Point 1
	points.push_back(Point3f(length, length, 0.f)); // Point 2
	points.push_back(Point3f(0.f, length, 0.f)); // Point 3

	points.push_back(Point3f(0.f, 0.f, -length)); // Point 4
	points.push_back(Point3f(length, 0.f, -length)); // Point 5
	points.push_back(Point3f(length, length, -length)); // Point 6
	points.push_back(Point3f(0.f, length, -length)); // Point 7

	// Projects points to world space using projectPoints method.
	projectPoints(points, rVectors, tVectors, cameraMat, distCoef, projectedPoints);

	// Create lines from cube points
	line(image, projectedPoints[0], projectedPoints[1], color, thickness);
	line(image, projectedPoints[1], projectedPoints[2], color, thickness);
	line(image, projectedPoints[2], projectedPoints[3], color, thickness);
	line(image, projectedPoints[3], projectedPoints[0], color, thickness);

	line(image, projectedPoints[4], projectedPoints[5], color, thickness);
	line(image, projectedPoints[5], projectedPoints[6], color, thickness);
	line(image, projectedPoints[6], projectedPoints[7], color, thickness);
	line(image, projectedPoints[7], projectedPoints[4], color, thickness);

	line(image, projectedPoints[0], projectedPoints[4], color, thickness);
	line(image, projectedPoints[1], projectedPoints[5], color, thickness);
	line(image, projectedPoints[2], projectedPoints[6], color, thickness);
	line(image, projectedPoints[3], projectedPoints[7], color, thickness);

}

// Method for calibration using the webcam. 
int liveCalibration(Mat frame, Mat drawToFrame, vector<Mat> savedImages, Mat cameraMat, Mat distanceCoefficients)
{
	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return -1;
	}

	namedWindow("Webcam", WINDOW_AUTOSIZE);

	//Finds the chessboard pattern from the camera
	while (true)
	{
		// If the camera is not giving back any input abort. 
		if (!vid.read(frame))
			break;

		vector<Vec2f> foundPoints;
		bool found = false;
		found = findChessboardCorners(frame, boardDim, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);


		if (!cameraCalibrated) {
			drawChessboardCorners(drawToFrame, boardDim, foundPoints, found);
		}

		if (found) {
			imshow("Webcam", drawToFrame);

			if (cameraCalibrated)
			{
				vector<Point3f> worldCornerPoints;
				createKnowBoardPositions(boardDim, squareDim, worldCornerPoints);

				// Finds an object pose from 3D-2D point correspondences, 
				// Calculates the rotation and translation vector
				Mat rVectors, tVectors;
				solvePnP(worldCornerPoints, foundPoints, cameraMat, distanceCoefficients, rVectors, tVectors);


				// Draws the coordinate axes
				drawAxis(0.1f, 0.0f, 0.0f, RED, rVectors, tVectors, cameraMat, distanceCoefficients, frame);
				drawAxis(0.0f, 0.1f, 0.0f, GREEN, rVectors, tVectors, cameraMat, distanceCoefficients, frame);
				drawAxis(0.0f, 0.0f, 0.1f, BLUE, rVectors, tVectors, cameraMat, distanceCoefficients, frame);

				// Draw a cube from the origin
				drawCube(0.05f, 2, WHITE, rVectors, tVectors, cameraMat, distanceCoefficients, frame);

			}
			else
			{
				if (savedImages.size() >= minSavedImages)
				{
					putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + "/" + to_string(minSavedImages) + ". Press Enter to calibrate.",
						cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
				}
				else
				{
					putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + "/" + to_string(minSavedImages) + ".",
						cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
				}
			}
			imshow("Webcam", frame);
		}
		else
		{
			if (!cameraCalibrated)
			{
				if (savedImages.size() >= 15)
				{
					putText(frame, "Press L to load saved calibration, use chessboard to calibrate,",
						cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);

					putText(frame, "or press Enter to calibrate.",
						cv::Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
				}
				else
				{
					putText(frame, "Press L to load saved calibration or use chessboard to calibrate.",
						cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
				}
			}

			imshow("Webcam", frame);
		}

		//Input handling
		char character = waitKey(1000 / framePerSecond);

		switch (character)
		{
		case ' ':// Spacebar
			//save image for calibration

			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:// Enter
			//Start calibration
			// If you saved 15 images at least with space you can start calibration
			if (savedImages.size() > minSavedImages)
			{
				cameraCalibration(savedImages, boardDim, squareDim, cameraMat, distanceCoefficients);
				saveCamCalibration("Calibration", cameraMat, distanceCoefficients);
				cameraCalibrated = true;
			}
			break;
		case 'l':
			// Load camera calibration data from file
			loadCameraCalibration("Calibration", cameraMat, distanceCoefficients);
			cameraCalibrated = true;
			break;
		case 27: //Escape
			// exit
			return 0;
			break;
		}
	}
	return 0;
}

// Main which starts the webcam and finds the corners on the checkerboard image and creates the camera matrix.
int main(int argv, char** argc)
{
	Mat frame;
	Mat drawToFrame;
	Mat cameraMat = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;
	vector<Mat> savedImages;
	vector<vector<Point2f>> markerCorners, rejectedCanidates;

	cout << "Press i for images for intrinsic calibration and v for live video calibration or \nl to load a calibration, then press enter to start.  \n";
	char ch = getchar();

	if (ch == 'i')
	{
		vector<cv::String> fn;
		glob("C:/Users/Lisa/Pictures/Camera Roll/*.jpg", fn, false);

		size_t count = fn.size(); //number of png files in images folder
		for (size_t i = 0; i < minSavedImages; i++)
			savedImages.push_back(imread(fn[i]));

		cout << "Images loaded \n";
		cout << "Starting Camera Calibration. \n";
		cameraCalibration(savedImages, boardDim, squareDim, cameraMat, distanceCoefficients);
		cout << "Camera calibration complete. \n";
		saveCamCalibration("CalibrationValues", cameraMat, distanceCoefficients);

		cout << "Starting webcam for cube drawing. \n";

		VideoCapture vid(0);

		if (!vid.isOpened())
		{
			return -1;
		}

		namedWindow("Webcam", WINDOW_AUTOSIZE);

		while (true)
		{
			// If the camera is not giving back any input abort. 
			if (!vid.read(frame))
				break;

			vector<Vec2f> foundPoints;
			bool found = false;
			found = findChessboardCorners(frame, boardDim, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
			frame.copyTo(drawToFrame);


			if (!cameraCalibrated) {
				drawChessboardCorners(drawToFrame, boardDim, foundPoints, found);
			}

			if (found) {
				imshow("Webcam", drawToFrame);

				if (cameraCalibrated)
				{
					vector<Point3f> worldCornerPoints;
					createKnowBoardPositions(boardDim, squareDim, worldCornerPoints);

					// Finds an object pose from 3D-2D point correspondences, 
					// Calculates the rotation and translation vector
					Mat rVectors, tVectors;
					solvePnP(worldCornerPoints, foundPoints, cameraMat, distanceCoefficients, rVectors, tVectors);


					// Draws the coordinate axes
					drawAxis(0.1f, 0.0f, 0.0f, RED, rVectors, tVectors, cameraMat, distanceCoefficients, frame);
					drawAxis(0.0f, 0.1f, 0.0f, GREEN, rVectors, tVectors, cameraMat, distanceCoefficients, frame);
					drawAxis(0.0f, 0.0f, 0.1f, BLUE, rVectors, tVectors, cameraMat, distanceCoefficients, frame);

					// Draw a cube from the origin
					drawCube(0.05f, 2, WHITE, rVectors, tVectors, cameraMat, distanceCoefficients, frame);

				}
				else
				{
					if (savedImages.size() >= minSavedImages)
					{
						putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + "/" + to_string(minSavedImages) + ". Press Enter to calibrate.",
							cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
					}
					else
					{
						putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + "/" + to_string(minSavedImages) + ".",
							cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
					}
				}
				imshow("Webcam", frame);
			}
			else
			{
				if (!cameraCalibrated)
				{
					putText(frame, "Press L to load saved calibration or escape to exit.",
					cv::Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200, 200, 250), 1, LINE_AA);
				}

				imshow("Webcam", frame);
			}

			//Input handling
			char character = waitKey(1000 / framePerSecond);

			switch (character)
			{
			case 'l':
				// Load camera calibration data from file
				loadCameraCalibration("Calibration", cameraMat, distanceCoefficients);
				cameraCalibrated = true;
				break;
			case 27: //Escape
				// exit
				return 0;
				break;
			}
		}
    }

	if (ch == 'v')
	{
		liveCalibration(frame, drawToFrame, savedImages, cameraMat, distanceCoefficients);
	}
}
