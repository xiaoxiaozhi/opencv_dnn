#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	string bin_model = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.caffemodel";
	string protxt = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.prototxt";

	// load DNN model
	Net net = readNetFromCaffe(protxt, bin_model);

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 获取各层信息
	vector<string> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		printf("layer id : %d, type : %s, name : %s \n", id, layer->type.c_str(), layer->name.c_str());
	}
	Mat src = imread("D:/images/paiqiu.png");
	imshow("input", src);

	// 构建输入
	Mat blob = blobFromImage(src, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false, false);
	net.setInput(blob, "data");

	// 执行推理
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;

	// 解析输出数据
	for (int i = 0; i < detectionMat.rows; i++) {
		float score = detectionMat.at<float>(i, 2);
		if (score > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * src.cols;
			float tl_y = detectionMat.at<float>(i, 4) * src.rows;
			float br_x = detectionMat.at<float>(i, 5)* src.cols;
			float br_y = detectionMat.at<float>(i, 6)* src.rows;
			Rect box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(src, box, Scalar(0, 0, 255), 2, 8, 0);
			putText(src, format("score: %.2f, %s", score, objNames[objIndex].c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2, 8);
		}
	}
	imshow("ssd-detection-demo", src);
	waitKey(0);
	return 0;
}