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

#include <stdlib.h>
#include <string.h>

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib, "libfacedetect-x64.lib")

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

int main(int argc, char *argv[])
{
	char imageRootDir[512] = "";
	int doLandmark = 0;
	int showResult = 1;
	int saveResult = 0;

	int root_len = 0;

	printf("\nThis application is to detect faces and landmarks in the input image!\n");
	printf("Usage: %s <image_list_name>\n"
			"    [--root <image_root_dir=''>]\n"
			"    [--output <output_file_name='./fd_rlt.txt'>]\n"
			"    [--landmark [<0 or 1>=0]]\n"
			"    [--show [<0 or 1>=1]]\n"
			"    [--save [<0 or 1>=0]]\n",
			argv[0]);

	if (argc < 2)
	{
		return -1;
	}

	FILE *fp = fopen(argv[1], "r");

	//load an image and convert it to gray (single-channel)
	if (fp == NULL)
	{
		fprintf(stderr, "Can not open image list file %s.\n", argv[1]);
		return -1;
	}

	char line_buf[256];

	char image_fn[512];
	char save_fn[512];
	int line_cnt = 0;

	char output_fn[512] = "./fd_rlt.txt";

	if (argc > 2)
	{
		for (int i = 2; i < argc; i++)
		{
			if (strncmp(argv[i], "--output", 8) == 0)
			{
				if (i + 1 < argc && argv[i + 1][0] != '-')
				{
					strcpy_s(output_fn, argv[i + 1]);
					i += 1;
				}
				else
				{
					fprintf(stderr, "Invalid arguments.");
					return -1;
				}
			}
			else if (strncmp(argv[i], "--root", 6) == 0)
			{
				if (i + 1 < argc && argv[i + 1][0] != '-')
				{
					strcpy_s(imageRootDir, argv[i + 1]);
					i += 1;
				}
				else
				{
					fprintf(stderr, "Invalid arguments.");
					return -1;
				}
			}
			else if (strncmp(argv[i], "--landmark", 10) == 0)
			{
				if (i + 1 < argc && argv[i + 1][0] != '-')
				{
					doLandmark = argv[i + 1][0] > '0';
					i += 1;
				}
				else
				{
					doLandmark = 1;
				}
			}
			else if (strncmp(argv[i], "--show", 6) == 0)
			{
				if (i + 1 < argc && argv[i + 1][0] != '-')
				{
					showResult = argv[i + 1][0] > '0';
					i += 1;
				}
				else
				{
					showResult = 1;
				}
			}
			else if (strncmp(argv[i], "--save", 6) == 0)
			{
				if (i + 1 < argc && argv[i + 1][0] != '-')
				{
					saveResult = argv[i + 1][0] > '0';
					i += 1;
				}
				else
				{
					saveResult = 1;
				}
			}
		}
	}

	root_len = strlen(imageRootDir);

	FILE *fp_out = fopen(output_fn, "w");

	//load an image and convert it to gray (single-channel)
	if (fp_out == NULL)
	{
		fprintf(stderr, "Can not open output file file %s.\n", output_fn);
		return -1;
	}

	int *pResults = NULL;
	//pBuffer is used in the detection functions.
	//If you call functions in multiple threads, please create one buffer for each thread!
	unsigned char *pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}

	while ((fgets(line_buf, 512, fp)) != NULL)
	{
		int len = strlen(line_buf);
		int n_faces = -1;

		line_cnt++;

		printf("\n===> Process line %d: %s\n", line_cnt, line_buf);

		if (len < 3 || line_buf[0] == '#')
		{
			fprintf(stderr, "Skip empty line or line starting with '#'.\n");
			continue;
		}

		line_buf[len - 1] = '\0';

		fprintf(fp_out, "%s\t", line_buf);

		if (root_len > 0)
		{
			sprintf(image_fn, "%s/%s", imageRootDir, line_buf);
		}
		else
		{
			strcpy_s(image_fn, line_buf);
		}

		Mat image = imread(image_fn);

		if (image.empty())
		{
			fprintf(stderr, "Failed to read image: %s.\n", image_fn);
			fprintf(fp_out, "%d\n", n_faces);

			continue;
		}

		Mat gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		///////////////////////////////////////////
		// reinforced multiview face detection / 68 landmark detection
		// it can detect side view faces, better but slower than facedetect_multiview().
		//////////////////////////////////////////
		//!!! The input image must be a gray one (single-channel)
		//!!! DO NOT RELEASE pResults !!!
		pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char *)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
												  1.2f, 2, 48, 0, doLandmark);

		printf("%d faces detected.\n", (pResults ? *pResults : 0));

		n_faces = pResults ? *pResults : 0;
		fprintf(fp_out, "%d\n", n_faces);

		Mat result_multiview_reinforce = image;

		//print the detection results
		for (int i = 0; i < n_faces; i++)
		{
			short *p = ((short *)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];

			printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
			fprintf(fp_out, "face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);

			if (showResult || saveResult)
			{
				rectangle(result_multiview_reinforce, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
				if (doLandmark)
				{
					for (int j = 0; j < 68; j++)
					{
						circle(result_multiview_reinforce, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
					}
				}
			}
		}

		if (showResult)
		{
			imshow("Results_multiview_reinforce", result_multiview_reinforce);
			//imwrite("Results_multiview_reinforce.jpg", result_multiview_reinforce);
			waitKey();
		}

		if (saveResult)
		{
			sprintf(save_fn, "%s_fd_rlt.jpg", image_fn);
			imwrite(save_fn, result_multiview_reinforce);
		}

		//release the buffer
		fflush(fp_out);
	}

	free(pBuffer);

	fclose(fp);
	fclose(fp_out);

	return 0;
}