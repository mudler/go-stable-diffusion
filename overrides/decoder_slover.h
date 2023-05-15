#pragma once
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
using namespace std;

class DecodeSlover
{
public:
    DecodeSlover(int h, int w, string assets_dir);

    ncnn::Mat decode(ncnn::Mat sample);

private:
    void generate_param(int height, int width, string assets_dir);

    const float factor[4] = { 1.0 / 0.18215f, 1.0 / 0.18215f, 1.0 / 0.18215f, 1.0 / 0.18215f };

    const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
    const float _norm_[3] = { 127.5f, 127.5f, 127.5f };

    ncnn::Net net;
};