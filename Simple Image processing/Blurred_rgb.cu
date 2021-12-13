#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ float gaussianKernelDevice[256];

__global__ void convolution(int rows, int cols, int kRows, int kCols, unsigned char* input, unsigned char* output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int pWidth = kCols / 2;
    int pHeight = kRows / 2;
    float sum = 0.0;

    if (idx >= pWidth && idx < cols - pWidth && idy < rows - pHeight && idy >= pHeight) {
        for (int i = 0; i < kRows; i++) {
            for (int j = 0; j < kCols; j++) {
                sum += gaussianKernelDevice[j + i * kCols] * input[(idx + j - pWidth) + (idy + i - pHeight) * cols];
            }
        }
        output[idx + idy * cols] = (unsigned char)sum;
    }
}

__global__ void SeparateChannels(int rows, int cols, unsigned char* rgb_image, unsigned char* r, unsigned char* g, unsigned char* b){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    const int channels = 3;

    if (idx < cols && idy < rows) {
        r[idx + idy*cols] = rgb_image[(idx + idy*cols) * channels];
        g[idx + idy*cols] = rgb_image[(idx + idy*cols) * channels + 1];
        b[idx + idy*cols] = rgb_image[(idx + idy*cols) * channels + 2];
    }
}

__global__ void CombineChannles(int rows, int cols, unsigned char* rgb_image, unsigned char* r, unsigned char* g, unsigned char* b) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    const int channels = 3;

    if (idx < cols && idy < rows) {
        rgb_image[(idx + idy * cols) * channels] = r[idx + idy * cols];
        rgb_image[(idx + idy * cols) * channels + 1] = g[idx + idy * cols];
        rgb_image[(idx + idy * cols) * channels + 2] = b[idx + idy * cols];
    }
}

int main(int argc, char** argv) {
    // Open a webcamera
    cv::VideoCapture camera(0);
    cv::Mat frame;
    if (!camera.isOpened()) return -1;

    // Create the cuda event timers 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    camera >> frame;
    cv::Mat output(frame.size().height, frame.size().width, CV_8UC3);
    output = 0; //Initialize the cv::Mat with zeros, so it can be overwrited with cudaMat data at the end

    const float gaussianKernel5x5[25] =
    {
        1.f / 273.f,  4.f / 273.f,  7.f / 273.f,  4.f / 273.f, 1.f / 273.f,
        4.f / 273.f,  16.f / 273.f, 26.f / 273.f,  16.f / 273.f, 4.f / 273.f,
        7.f / 273.f, 26.f / 273.f, 41.f / 273.f, 26.f / 273.f, 7.f / 273.f,
        1.f / 273.f,  4.f / 273.f,  7.f / 273.f,  4.f / 273.f, 1.f / 273.f,
        4.f / 273.f,  16.f / 273.f, 26.f / 273.f,  16.f / 273.f, 4.f / 273.f,
    };
    cudaMemcpyToSymbol(gaussianKernelDevice, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);

    unsigned char* cuda_input = NULL;
    unsigned char* cuda_output = NULL;
    unsigned char* cuda_r = NULL;
    unsigned char* cuda_g = NULL;
    unsigned char* cuda_b= NULL;
    unsigned char* cuda_r_conv = NULL;
    unsigned char* cuda_g_conv = NULL;
    unsigned char* cuda_b_conv = NULL;

    cudaMalloc(&cuda_input, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels());
    cudaMalloc(&cuda_output, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels());
    cudaMalloc(&cuda_r, sizeof(unsigned char) * frame.size().width * frame.size().height);
    cudaMalloc(&cuda_g, sizeof(unsigned char) * frame.size().width * frame.size().height);
    cudaMalloc(&cuda_b, sizeof(unsigned char) * frame.size().width * frame.size().height);
    cudaMalloc(&cuda_r_conv, sizeof(unsigned char) * frame.size().width * frame.size().height);
    cudaMalloc(&cuda_g_conv, sizeof(unsigned char) * frame.size().width * frame.size().height);
    cudaMalloc(&cuda_b_conv, sizeof(unsigned char) * frame.size().width * frame.size().height);

    // Loop while capturing images
    while (1)
    {
        // Capture the image and store a gray conversion to the gpu
        camera >> frame;
        cudaMemcpy(cuda_input, frame.data, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels(), cudaMemcpyHostToDevice);
        cudaMemset(cuda_output, 0, sizeof(unsigned char) * frame.size().height * frame.size().width);
        cudaMemset(cuda_r_conv, 0, sizeof(unsigned char) * frame.size().height * frame.size().width);
        cudaMemset(cuda_g_conv, 0, sizeof(unsigned char) * frame.size().height * frame.size().width);
        cudaMemset(cuda_b_conv, 0, sizeof(unsigned char) * frame.size().height * frame.size().width);
        // Record the time it takes to process
        cudaEventRecord(start);
        {
            // convolution kernel launch parameters
            dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
            dim3 cthreads(16, 16);
            
            SeparateChannels << <cblocks, cthreads >> > (frame.size().height, frame.size().width,cuda_input,cuda_r,cuda_g,cuda_b);
            
            convolution << < cblocks, cthreads >> > (frame.size().height, frame.size().width, 5, 5, cuda_r, cuda_r_conv);
            convolution << < cblocks, cthreads >> > (frame.size().height, frame.size().width, 5, 5, cuda_g, cuda_g_conv);
            convolution << < cblocks, cthreads >> > (frame.size().height, frame.size().width, 5, 5, cuda_b, cuda_b_conv);

            CombineChannles << < cblocks, cthreads >> > (frame.size().height,frame.size().width,cuda_output,cuda_r_conv,cuda_g_conv,cuda_b_conv);
            cudaMemcpy(output.data, cuda_output, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels(), cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop);

        // Display the elapsed time
        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

        // Show the results
        char line[99];
        sprintf_s(line, 99, "FPS: %f", 1000 / ms);
        cv::putText(output, line, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, false);
        cv::imshow("Original", frame);
        cv::imshow("Result", output);
        // Spin
        if (cv::waitKey(1) == 27) break;
    }

    // Exit
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    cudaFree(cuda_r);
    cudaFree(cuda_g);
    cudaFree(cuda_b);
    cudaFree(cuda_r_conv);
    cudaFree(cuda_g_conv);
    cudaFree(cuda_b_conv);

    return 0;
}