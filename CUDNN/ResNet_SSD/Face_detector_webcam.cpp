#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

VideoCapture camera(0);

int main(int argc, char** argv)
{

    // Setup variables and matrices
    string file_path = "YOUR_PATH_TO_THE_FILES_FOLDER";

    auto net = readNetFromCaffe(file_path + "res10_300x300_ssd_iter_140000.prototxt",file_path + "res10_300x300_ssd_iter_140000.caffemodel");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    const float min_confidence_score = 0.6;

    while (camera.isOpened()) {
        Mat frame;
        bool isSucces = camera.read(frame);

        if (!isSucces) {
            cout << "Could not read the camera" << endl;
            break;
        }

        int image_height = frame.cols;
        int image_width = frame.rows;

        auto start = getTickCount();

        Mat blob = blobFromImage(frame,1.0,Size(300,300),Scalar(104.0,177.0,123.0),false,false);

        net.setInput(blob);
        Mat output = net.forward();

        auto end = getTickCount();

        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < results.rows; i++) {
            float confidence = results.at<float>(i, 2);

            // Check if the detection is over the min threshold and then draw bbox
            if (confidence > min_confidence_score) {
                int bboxX = int(results.at<float>(i,3) * image_height);
                int bboxY = int(results.at<float>(i, 4) * image_width);
                int bboxWidth = int(results.at<float>(i, 5) * image_height - bboxX);
                int bboxHeight = int(results.at<float>(i, 6) * image_width - bboxY);
                rectangle(frame, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0, 0, 255), 2);
                string class_name = "Face";
                putText(frame, class_name + " " + to_string(int(confidence * 100)) + "%", Point(bboxX, bboxY - 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
            }
        }


        auto totalTime = (end - start) / getTickFrequency();


        putText(frame, "FPS: " + to_string(int(1 / totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);

        imshow("image", frame);

        if (waitKey(1) == 'q') {
            break;
        }

    }

    camera.release();
    destroyAllWindows();

    return 0;
}