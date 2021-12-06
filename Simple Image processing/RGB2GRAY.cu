#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel(int rows, int cols, int channels, unsigned char* input, unsigned char *output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx < cols && idy < rows) {
        unsigned char r = input[(idx + idy * cols)*channels];
        unsigned char g = input[(idx + idy * cols)*channels + 1];
        unsigned char b = input[(idx + idy * cols)*channels + 2];
        output[idx + idy * cols] = r * 0.299f + g * 0.587f + b * 0.114f;
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
    cv::Mat source(frame.size().height,frame.size().width,CV_8UC1);
    source = 0; //Initialize the cv::Mat with zeros, so it can be overwrited with cudaMat data

    unsigned char* cuda_source = NULL;
    unsigned char* cuda_output = NULL;
    cudaMalloc(&cuda_source, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels());
    cudaMalloc(&cuda_output, sizeof(unsigned char) * frame.size().width * frame.size().height);
    // Loop while capturing images
    while (1)
    {
        // Capture the image and store a gray conversion to the gpu
        camera >> frame;
        cudaMemcpy(cuda_source, frame.data, sizeof(unsigned char) * frame.size().width * frame.size().height * frame.channels(), cudaMemcpyHostToDevice);
        cudaMemset(cuda_output, 0, sizeof(unsigned char) * frame.size().height * frame.size().width);
        // Record the time it takes to process
        cudaEventRecord(start);
        {
            // convolution kernel launch parameters
            dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
            dim3 cthreads(16, 16);
            kernel << < cblocks, cthreads >> > (frame.size().height, frame.size().width, frame.channels(), cuda_source, cuda_output);
            cudaMemcpy(source.data, cuda_output, sizeof(unsigned char) * frame.size().width * frame.size().height, cudaMemcpyDeviceToHost);
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
        cv::putText(source, line, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, false);
        cv::imshow("Original", frame);
        cv::imshow("Result", source);
        // Spin
        if (cv::waitKey(1) == 27) break;
    }

    // Exit
    cudaFree(cuda_source);
    cudaFree(cuda_output);

    return 0;
}