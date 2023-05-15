#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <cmath>
#include <ncnn/src/net.h>
#include "prompt_slover.h"
#include "decoder_slover.h"
#include "diffusion_slover.h"
#include "diffusion_slover.cpp"
#include "decoder_slover.cpp"
#include "prompt_slover.cpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <ctime>
#include "getmem.h"
#include "stablediffusion.h"

using namespace std;

int generate_image(int height, int width, int mode, int  step, int seed, const char *positive_prompt, const char *negative_prompt, const char *dst)
{

	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	

	// stable diffusion
	cout << "----------------[init]--------------------";
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover(height, width, mode);
	DecodeSlover decode_slover(height, width);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);
std::string positive_prompt_str = positive_prompt;
std::string negative_prompt_str = negative_prompt;
	cout << "----------------[prompt]------------------";
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt_str);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt_str);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[diffusion]---------------" << endl;
	ncnn::Mat sample = diffusion_slover.sampler(seed, step, cond, uncond);
	cout << "----------------[diffusion]---------------";
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[decode]------------------";
	ncnn::Mat x_samples_ddim = decode_slover.decode(sample);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[save]--------------------" << endl;
	cv::Mat image(height, width, CV_8UC3);
	x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
	//cv::imwrite(dst, image);
	cv::imwrite(dst, image);

	cout << "----------------[close]-------------------" << endl;
	
	return 0;
}
