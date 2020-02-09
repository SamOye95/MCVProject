#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const float squareDim = 0.023f; // meters
//const float arucoSquareDim = 
const Size boardDim = Size(6, 9);

void createArucoMarkers()
{
	Mat outputMarker;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	
	for(int i =0; i < 50; i++)
	{
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "4x4_Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
	}
}

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

void getCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf; 
		bool found = findChessboardCorners(*iter, boardDim, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		if (found)
			allFoundCorners.push_back(pointBuf);
		if (showResults)
		{
			drawChessboardCorners(*iter, boardDim, pointBuf, found);
			imshow("Corner detection", *iter);
			waitKey();
		}
	}
}

void cameraCalibration(vector<Mat> calImages, Size boardSize, float edgeLen, Mat& camMat, Mat& distCoef)
{
	vector<vector<Point2f>> imgSpacePoints;
	getCorners(calImages, imgSpacePoints, false);

	vector<vector<Point3f>> worldCornerPoints(1); 

	createKnowBoardPositions(boardSize, edgeLen, worldCornerPoints[0]);
	worldCornerPoints.resize(imgSpacePoints.size(), worldCornerPoints[0]);

	vector<Mat> rVectors, tVectors; 
	distCoef = Mat::zeros(8, 1, CV_64F); 

	calibrateCamera(worldCornerPoints, imgSpacePoints, boardSize, camMat, distCoef, rVectors , tVectors);



}

bool saveCamCalibration(string filename, Mat cameraMat, Mat distCoef)
{
	ofstream outStream(filename);
	if (outStream)
	{
		uint16_t rows = cameraMat.rows;
		uint16_t cols = cameraMat.cols; 

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = cameraMat.at<double>(r, c);
				outStream << value << endl; 
			}
		}
		rows = distCoef.rows; 
		cols = distCoef.cols; 

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = distCoef.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;

	}
	return false; 
}

int main(int argv, char** argc)
{
	Mat frame;
	Mat drawToFrame;
	Mat cameraMat = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients; 
	vector<Mat> savedImages; 
	vector<vector<Point2f>> markerCorners, rejectedCanidates;
	
	VideoCapture vid(0);

	if (!vid.isOpened())
		return -1;

	int framePerSecond = 20; 
	namedWindow("Webcam", WINDOW_AUTOSIZE);

	while (true)
	{
		if (!vid.read(frame))
			break; 

		vector<Vec2f> foundPoints;
		bool found = false;
		found = findChessboardCorners(frame, boardDim, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, boardDim, foundPoints, found);
		if (found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);

		char character = waitKey(1000 / framePerSecond);

		switch (character)
		{
		case ' ':// Spacebar
			//save image
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:// Enter
			//Start calibration
			if (savedImages.size() > 15)
			{
				cameraCalibration(savedImages, boardDim, squareDim, cameraMat, distanceCoefficients);
				saveCamCalibration("Calibration", cameraMat, distanceCoefficients);
			}
			break;
		case 27: //Escape
			// exit
			return 0; 
			break;
		}
	}
}