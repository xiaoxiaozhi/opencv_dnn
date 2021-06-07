#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<Vec3b> colors_table;
void colorizeSegmentation(const Mat &score, Mat &segm);
int main(int argc, char** argv) {
	string enet_model = "D:/projects/opencv_tutorial/data/models/enet/model-best.net";

	// load network model
	Net net = readNetFromTorch(enet_model);

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
	Mat src = imread("D:/projects/models/enet/test.png");
	imshow("input", src);

	// 构建输入
	Mat blob = blobFromImage(src, 0.00392, Size(512, 256), Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	// 执行推理
	Mat scores = net.forward();
	
	// 解析输出
	Mat segm;
	colorizeSegmentation(scores, segm);

	resize(segm, segm, src.size());
	addWeighted(src, 0.5, segm, 0.5, 0, src);
	imshow("enet-segmentation-demo", src);
	waitKey(0);
	return 0;
}

void colorizeSegmentation(const Mat &score, Mat &segm) {
	const int rows = score.size[2];
	const int cols = score.size[3];
	const int cns = score.size[1];
	printf("height: %d, width: %d, channels : %d \n", rows, cols, cns);
	// generator color table
	colors_table.push_back(Vec3b());
	for (int i = 1; i < cns; i++) {
		Vec3b color;
		for (int j = 0; j < 3; j++) {
			color[j] = (colors_table[i - 1][j] + rand() % 256) / 2;
		}
		colors_table.push_back(color);
	}

	// 解析通道数据
	Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
	Mat maxVal(rows, cols, CV_32FC1, score.data);
	for (int ch = 1; ch < cns; ch++) {
		for (int row = 0; row < rows; row++) {
			const float *prtScore = score.ptr<float>(0, ch, row);
			float *ptrMaxval = maxVal.ptr<float>(row);
			uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
			for (int col = 0; col < cols; col++) {
				if (prtScore[col] > ptrMaxval[col]) {
					ptrMaxval[col] = prtScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}

	segm.create(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++) {
		Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
		const uchar* prtMaxCl = maxCl.ptr<uchar>(row);
		for (int col = 0; col < cols; col++) {
			ptrSegm[col] = colors_table[prtMaxCl[col]];
		}
	}
}