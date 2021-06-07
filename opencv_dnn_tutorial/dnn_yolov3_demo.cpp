#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;
string label_map = "D:/projects/opencv_tutorial/data/models/object_detection_classes_yolov3.txt";
string yolov3_tiny_weight = "D:/projects/opencv_tutorial/data/models/yolov3-tiny-coco/yolov3-tiny.weights";
string yolov3_tiny_cfg = "D:/projects/opencv_tutorial/data/models/yolov3-tiny-coco/yolov3-tiny.cfg";
int main(int argc, char** argv) {
	string weight_file = "D:/projects/models/yolov3/yolov3.weights";
	string cfg_file = "D:/projects/models/yolov3/yolov3.cfg";

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

	// load network model
	Net net = readNetFromDarknet(yolov3_tiny_cfg, yolov3_tiny_weight);
	vector<string> outNames = net.getUnconnectedOutLayersNames();
	for (int k = 0; k < outNames.size(); k++) {
		printf("output layer name : %s \n", outNames[k].c_str());
	}

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

	// 设置输入数据
	Mat blob = blobFromImage(src, 0.00392, Size(416, 416), Scalar(), true, false);
	net.setInput(blob);

	// 推理输出
	vector<Mat> outs;
	net.forward(outs, layer_names);
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (int i = 0; i < outs.size(); i++) {
		// 开始解析每个输出blob
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols) {
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5) {
				int centerx = (int)(data[0] * src.cols);
				int centery = (int)(data[1] * src.rows);
				int width = (int)(data[2] * src.cols);
				int height = (int)(data[3] * src.rows);
				int left = centerx - width / 2;
				int top = centery - height / 2;
				classIds.push_back(classIdPoint.x);
				confidences.push_back(confidence);
				boxes.push_back(Rect(left, top, width, height));
				// rectangle(src, Rect(left, top, width, height), Scalar(255, 0, 0), 1, 8);
			}
		}
	}
	vector<int> indexes;
	NMSBoxes(boxes, confidences, 0.5, 0.5, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = classIds[i];
		Rect box = boxes[i];
		rectangle(src, box, Scalar(0, 0, 255), 2, 8);
		putText(src, format("score: %.2f, %s", confidences[indexes[i]], classNames[idx].c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
	}
	imshow("YOLOv3-detection-demo", src);
	waitKey(0);
	return 0;
}