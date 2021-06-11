#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> readLabels();
//图像检测 SSD模型
int main(int argc, char** argv) {

	string bin_model = "./MobileNetSSD_deploy.caffemodel";//
	string protxt = "./MobileNetSSD_deploy.prototxt";

	// load DNN model  
	Net net = readNetFromCaffe(protxt, bin_model);

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 获取各层信息
	vector<string> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		//printf("layer id : %d, type : %s, name : %s \n", id, layer->type.c_str(), layer->name.c_str());
	}
	Mat src = imread("./xiaomaolu.jpg");
	imshow("input", src);
	vector<string> names = readLabels();

	// 构建输入 模型配置 https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml
	// 函数介绍https://www.zhihu.com/column/2know 
	int w = 300;
	int h = 300;
	Mat intpuBlob = blobFromImage(src, 0.007843, Size(w, h), Scalar(127.5, 127.5, 127.5), false, false);

	// 设置输入
	net.setInput(intpuBlob, "data");
	// 推断  函数返回一个Mat变量，返回值是指输入的layername首次出现的输出。
	//例如  输入 prob 就能返回该层的结果, 没有参数返回最后一层输出结果
	/*layer{
	name: "prob"
	type : "Softmax"
	bottom : "loss3/classifier"
	top : "prob"
	}*/
	//Mat first = net.forward("conv1/7x7_s2");//使用第一层输出的mat 没有宽高，跟讲解的不一样，存疑
	//cout << "channel = " << first.channels() << "\trows = " << first.rows << "\tcols = " << first.cols << endl;
	Mat probMat = net.forward();
	cout << "channel = " << probMat.channels() << "\trows = " << probMat.rows << "\tcols = " << probMat.cols << endl;

	//SSD 模型结果都存储在结构体size 中  
	Mat detectionOut(probMat.size[2], probMat.size[3], CV_32F, probMat.ptr<float>());
	cout << "detectionOut-channel = " << detectionOut.channels() << "\trows = " << detectionOut.rows << "\tcols = " << detectionOut.cols << endl;
	/*Point classNum;
	double classProb;
	minMaxLoc(prob, NULL,//最小值指针
		&classProb,//最大值指针
		NULL,//最小值位置指针
		&classNum);//最大值位置指针
	int index = classNum.x;
	printf("\n current index = %d, possible : %.2f , name : %s\n", index, classProb, names[index].c_str());
	putText(src, names[index].c_str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
	imshow("result", src);*/
	waitKey(0);
	return 0;
}

vector<string> readLabels() {
	string label_map_txt = "./labelmap_det.txt";
	vector<string> classNames;
	ifstream fp(label_map_txt);
	if (!fp.is_open()) {
		printf("could not find the file \n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name);
		}
	}
	fp.close();
	return classNames;
}