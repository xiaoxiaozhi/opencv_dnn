#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;
std::map<int, string> readLabelMaps();
string label_map = "D:/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt";
int main(int argc, char** argv) {
	string bin_model = "D:/tensorflow/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb";
	string pbtxt = "D:/tensorflow/faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco_2018_01_28.pbtxt";
	map<int, string> names = readLabelMaps();

	// load DNN model
	Net net = readNetFromTensorflow(bin_model, pbtxt);

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
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
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false);
	net.setInput(blob);

	// 执行推理
	Mat detection = net.forward();
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;

	// 解析输出数据
	for (int i = 0; i < detectionMat.rows; i++) {
		float score = detectionMat.at<float>(i, 2);
		if (score > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1)) + 1;
			float tl_x = detectionMat.at<float>(i, 3) * src.cols;
			float tl_y = detectionMat.at<float>(i, 4) * src.rows;
			float br_x = detectionMat.at<float>(i, 5)* src.cols;
			float br_y = detectionMat.at<float>(i, 6)* src.rows;
			Rect box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(src, box, Scalar(0, 0, 255), 2, 8, 0);
			map<int, string>::iterator it = names.find(objIndex);
			printf("id : %d, display name : %s \n", objIndex, (it->second).c_str());
			putText(src, format("score: %.2f, %s", score, (it->second).c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2, 8);
		}
	}
	imshow("faster-rcnn-detection-demo", src);
	waitKey(0);
	return 0;
}


std::map<int, string> readLabelMaps()
{
	std::map<int, string> labelNames;
	std::ifstream fp(label_map);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	string one_line;
	string display_name;
	while (!fp.eof())
	{
		std::getline(fp, one_line);
		std::size_t found = one_line.find("id:");
		if (found != std::string::npos) {
			int index = found;
			string id = one_line.substr(index + 4, one_line.length() - index);

			std::getline(fp, display_name);
			std::size_t  found = display_name.find("display_name:");

			index = found + 15;
			string name = display_name.substr(index, display_name.length() - index);
			name = name.replace(name.length() - 1, name.length(), "");
			// printf("id : %d, name: %s \n", stoi(id.c_str()), name.c_str());
			labelNames[stoi(id)] = name;
		}
	}
	fp.close();
	return labelNames;
}