/*
The MIT License (MIT)

Copyright (c) 2015-2017 Shiqi Yu
shiqi.yu@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetect-dll.h"

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib,"libfacedetect-x64.lib")

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

int main(int argc, char* argv[])
{
	int fd_type = 0;

    printf("Usage: %s [<FD_TYPE>]\n", argv[0]);
	printf(
		"<FD_TYPE>:\n"
		"			0 - facedetect_frontal, default\n"	
		"			1 - facedetect_frontal_surveillance\n"
		"			2 - facedetect_multiview\n"
		"			3 - facedetect_multiview_reinforce\n"
	);
	printf("Press ESC or 'q' to exit...");

	if (argc > 1)
	{
		sscanf(argv[1], "%d", &fd_type);
	}
	
	int doLandmark = 1;

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("Failed to open camera, exit!\n");
		return -1;
	}

	int * pResults = NULL;

	double t;
	double avg_time = 0.0f;
	double time_ttl = 0.0f;
	int frame_cnt = 0;

	//pBuffer is used in the detection functions.
	//If you call functions in multiple threads, please create one buffer for each thread!
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}

	while (1)
	{
		//load an image and convert it to gray (single-channel)
		Mat image;
		cap >> image;
		if (image.empty())
		{
			fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
			return -1;
		}
		Mat gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		t = (double)cvGetTickCount();
		switch (fd_type)
		{
		case 1:
			///////////////////////////////////////////
			// frontal face detection designed for video surveillance / 68 landmark detection
			// it can detect faces with bad illumination.
			//////////////////////////////////////////
			//!!! The input image must be a gray one (single-channel)
			//!!! DO NOT RELEASE pResults !!!
			pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
				1.2f, 2, 48, 0, doLandmark);
			break;
		case 2:
			///////////////////////////////////////////
			// multiview face detection / 68 landmark detection
			// it can detect side view faces, but slower than facedetect_frontal().
			//////////////////////////////////////////
			//!!! The input image must be a gray one (single-channel)
			//!!! DO NOT RELEASE pResults !!!
			pResults = facedetect_multiview(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
				1.2f, 2, 48, 0, doLandmark);
			break;
		case 3:
			///////////////////////////////////////////
			// reinforced multiview face detection / 68 landmark detection
			// it can detect side view faces, better but slower than facedetect_multiview().
			//////////////////////////////////////////
			//!!! The input image must be a gray one (single-channel)
			//!!! DO NOT RELEASE pResults !!!
			pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
				1.2f, 3, 48, 0, doLandmark);
			break;

		default:
			///////////////////////////////////////////
			// frontal face detection / 68 landmark detection
			// it's fast, but cannot detect side view faces
			//////////////////////////////////////////
			//!!! The input image must be a gray one (single-channel)
			//!!! DO NOT RELEASE pResults !!!
			pResults = facedetect_frontal(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
				1.2f, 2, 48, 0, doLandmark);
			break;
		}

		t = (double)cvGetTickCount() - t;
		t = t / ((double)cvGetTickFrequency()*1000.);

		frame_cnt += 1;
		time_ttl += t;
		avg_time = time_ttl / frame_cnt;

		printf("average time = %g ms; FPS = %.1f\n", avg_time, 1000.0 / avg_time);

		printf("%d faces detected.\n", (pResults ? *pResults : 0));
		Mat rlt_img = image.clone();
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];

			printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
			rectangle(rlt_img, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
			if (doLandmark)
			{
				for (int j = 0; j < 68; j++)
					circle(rlt_img, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
			}
		}
		imshow("Results_frontal", rlt_img);

		int k = waitKey(5);
		if (k == 27 || k == 'q' || k == 'Q')
		{
			break;
		}
	}

    //release the buffer
    free(pBuffer);

	return 0;
}