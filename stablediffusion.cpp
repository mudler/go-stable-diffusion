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

    /*
	int height, width, mode, step, seed;
	string positive_prompt, negative_prompt;

	// default setting
	height = 256;
	width = 256;
	mode = 0;
	step = 15;
	seed = 42;
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

	*/

    // parse the magic.txt
	ifstream magic;
	magic.open("magic.txt");
	if (!magic) {
		cout << "can not find magic.txt, using the default setting" << endl;
	}
	else {
		string content = "";
		int i = 0;
		for (i = 0; i < 7; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:height = stoi(content);
				case 1:width = stoi(content);
				case 2:mode = stoi(content);
				case 3:step = stoi(content);
				case 4:seed = stoi(content);
				case 5:positive_prompt = content.c_str();
				case 6:negative_prompt = content.c_str();
				default:break;
				}
			}
			else {
				break;
			}
		}
		if (i != 7) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}

	}

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
	cv::imwrite("result_" + to_string(step) + "_" + to_string(seed) + "_" + to_string(height) + "x" + to_string(width) + ".png", image);

	cout << "----------------[close]-------------------" << endl;
	
	return 0;
}
