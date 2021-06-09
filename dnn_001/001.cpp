#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> readLabels();
int main(int argc, char** argv) {

	string bin_model = "./bvlc_googlenet.caffemodel";//
	string protxt = "./bvlc_googlenet.prototxt";

	// load DNN model  
	Net net = readNetFromCaffe(protxt, bin_model);

	// ���ü����̨
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// ��ȡ������Ϣ
	vector<string> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		//printf("layer id : %d, type : %s, name : %s \n", id, layer->type.c_str(), layer->name.c_str());
	}
	Mat src = imread("./home.jpg");
	imshow("input", src);
	vector<string> names = readLabels();

	// �������� ģ������ https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml
	// ��������https://www.zhihu.com/column/2know 
	int w = 224;
	int h = 224;
	Mat intpuBlob = blobFromImage(src, 1.0, Size(w, h), Scalar(104, 117, 123), false, false);

	// ��������
	net.setInput(intpuBlob);
	// �ƶ�  ��������һ��Mat����������ֵ��ָ�����layername�״γ��ֵ������
	//����  ���� prob ���ܷ��ظò�Ľ��, û�в����������һ��������
	/*layer{
	name: "prob"
		  type : "Softmax"
		bottom : "loss3/classifier"
		top : "prob"
	}*/
	//Mat first = net.forward("conv1/7x7_s2");//ʹ�õ�һ�������mat û�п�ߣ�������Ĳ�һ��������
	//cout << "channel = " << first.channels() << "\trows = " << first.rows << "\tcols = " << first.cols << endl;
	Mat probMat = net.forward();
	cout << "channel = " << probMat.channels() << "\trows = " << probMat.rows << "\tcols = " << probMat.cols << endl;

	//��������
	Mat prob = probMat.reshape(1, 1);//���һά����

	Point classNum;
	double classProb;
	minMaxLoc(prob, NULL,//��Сֵָ��
		&classProb,//���ֵָ�� 
		NULL,//��Сֵλ��ָ��
		&classNum);//���ֵλ��ָ��
	int index = classNum.x;
	printf("\n current index = %d, possible : %.2f , name : %s\n", index, classProb, names[index].c_str());
	putText(src, names[index].c_str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
	imshow("result", src);
	waitKey(0);
	return 0;
}

vector<string> readLabels() {
	string label_map_txt = "./classification_classes_ILSVRC2012.txt";
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