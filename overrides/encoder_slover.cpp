#include "encoder_slover.h"
#include <filesystem>

EncodeSlover::EncodeSlover(int h, int w, string assets_dir)
{
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_bf16_storage = true;
	net.opt.use_packing_layout = true;

	// Define the name of the parameter file
	std::string param_file;
	if (h == 512 && w == 512)
		param_file = "AutoencoderKL-encoder-512-512-fp16.param";
	else
	{
		generate_param(h, w, assets_dir);
		param_file = "tmp-AutoencoderKL-encoder-" + to_string(h) + "-" + to_string(w) + "-fp16.param";
	}

	// Join the paths using std::filesystem::path::operator/() function
	std::filesystem::path param_path = std::filesystem::path(assets_dir) / std::filesystem::path(param_file);
	net.load_param(param_path.string().c_str());

	std::filesystem::path encoder_path = std::filesystem::path(assets_dir) / std::filesystem::path("AutoencoderKL-encoder-512-512-fp16.bin");

	net.load_model(encoder_path.string().c_str());

	h_size = h;
	w_size = w;
}

void EncodeSlover::generate_param(int height, int width, string assets_dir)
{
	string line;
	
	std::filesystem::path decoder = std::filesystem::path(assets_dir) / std::filesystem::path("assets/AutoencoderKL-encoder-512-512-fp16.param");
	std::filesystem::path decoder_out = std::filesystem::path(assets_dir) / std::filesystem::path("tmp-AutoencoderKL-encoder-" + std::to_string(height) + "-" + std::to_string(width) + "-fp16.param");

	ifstream encoder_file(decoder.string().c_str());
	ofstream encoder_file_new(decoder_out.string().c_str());

	int cnt = 0;
	while (getline(encoder_file, line))
	{
		if (line.substr(0, 7) == "Reshape")
		{
			switch (cnt)
			{
			case 0: line = line.substr(0, line.size() - 12) + "0=" + to_string(width * height / 8 / 8) + " 1=512"; break;
			case 1: line = line.substr(0, line.size() - 15) + "0=" + to_string(width / 8) + " 1=" + std::to_string(height / 8) + " 2=512"; break;
			default: break;
			}
			
			cnt++;
		}
		encoder_file_new << line << endl;
	}
	encoder_file_new.close();
	encoder_file.close();
}

std::vector<ncnn::Mat> EncodeSlover::encode(cv::Mat& bgr_image)
{
	std::vector<ncnn::Mat> mean_std(2);
	{
		int ih = bgr_image.rows, iw = bgr_image.cols;
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr_image.data, ncnn::Mat::PIXEL_BGR2RGB, iw, ih, w_size, h_size);
		in.substract_mean_normalize(_mean_, _norm_);

		{
			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.input("in0", in);
			ex.extract("out0", mean_std[0]);
			ex.extract("out1", mean_std[1]);
		}
	}

	return mean_std;
}