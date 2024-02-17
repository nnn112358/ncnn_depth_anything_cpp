#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "benchmark.h" // Benchmark utilities from NCNN
#include "net.h" // Main NCNN network handling
#include <iostream>

using namespace std;
using namespace cv;
using namespace ncnn;

// Function to perform detection using NCNN network and OpenCV images
int detect(const Net &dpt_, const Mat &rgb, Mat &depth_color) {
    int width = rgb.cols;
    int height = rgb.rows;

    // Calculate scale to pad image to a multiple of 32
    int w = width, h = height;
    float scale = 1.f;
    int target_size = 518;
    if (w > h) {
        scale = static_cast<float>(target_size) / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = static_cast<float>(target_size) / h;
        h = target_size;
        w = w * scale;
    }

    // Resize and pad the input image
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    
    // Normalize the image
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.01712475f, 0.0175f, 0.01742919f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    // Create extractor and perform inference
    ncnn::Extractor ex = dpt_.create_extractor();
    ncnn::Mat out;
    ex.input(dpt_.input_names()[0], in_pad);
    ex.extract(dpt_.output_names()[0], out);

    // Convert NCNN Mat to OpenCV Mat and apply colormap
    Mat depth(out.h, out.w, CV_32FC1, out.data);
    normalize(depth, depth, 0, 255, NORM_MINMAX, CV_8UC1);
    Mat color_map = Mat(target_size, target_size, CV_8UC3);
    applyColorMap(depth, color_map, COLORMAP_INFERNO);
    resize(color_map(Rect(wpad / 2, hpad / 2, w, h)), depth_color, rgb.size());

    return 0;
}

int main(int argc, char **argv) {
    // Load and prepare the input image
    string imagepath = "./test.jpg";
    if (argc >= 2) {
        imagepath = argv[1];
    }

    Mat input_img = imread(imagepath, IMREAD_COLOR);
    if (input_img.empty()) {
        cerr << "cv::imread failed for image: " << imagepath << endl;
        return -1;
    }

    // Initialize the NCNN network
    ncnn::Option opt;
    opt.use_vulkan_compute = true; // Enable Vulkan compute if available
    ncnn::Net net;
    net.opt = opt;

    // Load the network model and parameters
    const char *model_paths = "models/depth_anything_vits14.ncnn.bin";
    const char *param_paths = "models/depth_anything_vits14.ncnn.param";
    net.load_param(param_paths);
    net.load_model(model_paths);

    // Perform detection
    Mat depth_color;
    detect(net, input_img, depth_color);

    // Display results
    imshow("Input Image", input_img);
    imshow("Depth Color Map", depth_color);
    waitKey(0); // Wait for any key press

    net.clear(); // Clear the network to free resources

    return 0;
}

