#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel(int rows, int cols, unsigned char* input, int T) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx >= 0 && idy >= 0 && idx < cols && idy < rows) {
        if (input[idx + idy*cols] < T) input[idx + idy*cols] = 0;
        else input[idx + idy*cols] = 255;
    }
}

int main(int argc, char **argv) {
    // Open a webcamera
    cv::VideoCapture camera(0);
    cv::Mat frame;
    if (!camera.isOpened())
        return -1;

    // Ventanas para la captura
    cv::namedWindow("Original");
    cv::namedWindow("Binarized");

    // Create the cuda event timers 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    camera >> frame;

    int threshold = 90;
    
    cv::Mat source;
    unsigned char* cuda_source = NULL;
    cudaMalloc(&cuda_source, frame.size().width * frame.size().height);
    // Loop while capturing images
    while (1)
    {
        // Capture the image and store a gray conversion to the gpu
        camera >> frame;
        cv::cvtColor(frame, source, cv::COLOR_RGB2GRAY); //First, the RGB image is converted to GRAYSCALE
        cudaMemcpy(cuda_source, source.data, frame.size().width * frame.size().height,cudaMemcpyHostToDevice);
        // Record the time it takes to process
        cudaEventRecord(start);
        {
            // convolution kernel launch parameters
            dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
            dim3 cthreads(16, 16);
            kernel <<< cblocks, cthreads >>> (frame.size().height, frame.size().width, cuda_source, threshold);
            cudaMemcpy(source.data, cuda_source, sizeof(unsigned char) * frame.size().width * frame.size().height, cudaMemcpyDeviceToHost);
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
        sprintf_s(line, 99, "FPS: %f", 1000/ms);
        cv::putText(source, line, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(100, 45, 120), 1, false);
        cv::imshow("Original", frame);
        cv::imshow("Binarized", source);

        // Spin
        if (cv::waitKey(1) == 27) break;
    }

    // Exit
    cudaFree(cuda_source);

    return 0;
}