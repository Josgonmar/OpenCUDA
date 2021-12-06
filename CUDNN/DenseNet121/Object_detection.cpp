#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main() {
	string path_files = "YOUR PATH TO THE IMAGES FOLDER";
	vector<string> class_names;
	ifstream ifs(string(path_files + "MobileNet_SSD/object_detection_classes_coco.txt").c_str());
	string line;
	while (getline(ifs, line)) {
		class_names.push_back(line);
		//cout << line << endl;
	}
	auto model = readNet(path_files + "MobileNet_SSD/frozen_inference_graph.pb", path_files + "MobileNet_SSD/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", "TensorFlow");
	Mat input_image = imread(path_files + "IMAGE_TO_TRY.jpg");
	Mat blob = blobFromImage(input_image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);
	model.setInput(blob);
	Mat output = model.forward();
	Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++) {
		int class_id = detectionMat.at<float>(i, 1);
		float confidence = detectionMat.at<float>(i, 2);

		// Check if the detection is of good quality
		if (confidence > 0.4) {
			int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * input_image.cols);
			int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * input_image.rows);
			int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * input_image.cols - box_x);
			int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * input_image.rows - box_y);
			rectangle(input_image, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 255, 255), 2);
			putText(input_image, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
		}
	}

	imshow("image", input_image);
	waitKey(0);
	return 0;
}