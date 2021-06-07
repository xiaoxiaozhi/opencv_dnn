#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
	string pb_model = "D:/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
	string pb_txt = "D:/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/graph.pbtxt";
	string label_map = "D:/projects/opencv_tutorial/data/models/object_detection_classes_coco.txt";
	vector<string> classNames;
	ifstream fp(label_map);
	std::string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name);
		}
	}
	fp.close();

	// load DNN model
	Net net = readNetFromTensorflow(pb_model, pb_txt);

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
	Mat src = imread("D:/images/dog.jpg");
	imshow("input", src);

	// 构建输入
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	// 执行推理
	Mat detection = net.forward();
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
			putText(src, format("score: %.2f, %s", score, classNames[objIndex-1].c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2, 8);
		}
	}
	imshow("tensorflow-ssd-detection-demo", src); 
	waitKey(0); 
	return 0;
}