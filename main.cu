#include <iostream>
#include "network.cuh"
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>

using namespace std;

#define BATCH_SIZE 64
#define NUM_EPOCHS 100
#define NUM_TEST 4

vector<vector<float>> read_mnist();
vector<uint8_t> read_label();

int main() {
	vector<vector<float>> numbers = read_mnist();
	cout << "numbers loaded" << endl;
	vector<uint8_t> labels = read_label();
	cout << "label loaded" << endl;
	Network net(28, 1,0.1f);
	net.addConvLayer(7, 10, 1, false, reLu);
    net.addFullLayer(10, Sigmoid);
	float *out;
	float* sol = new float [10]();
	random_device r;
	uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, 59999);

	for (int j=0; j < NUM_EPOCHS; j++) {
		double loss = 0.0;
		for (int i = 0; i < BATCH_SIZE; i++) {
			int x = distribution(r);
			sol[labels[x]] = 1;
			out = net.forward(numbers[x].data());
			net.train(out, sol, numbers[x].data());
			for(int z=0; z<10; z++)
				loss += pow((out[z] - sol[z]), 2);
			sol[labels[x]] = 0;
		}
		delete[] out;
		net.learn();
		cout <<"loss: " << loss / BATCH_SIZE << endl;
	}
	exit(0);
	int hit = 0;
	for (int i = 0; i < NUM_TEST; i++) {
		int x = i;
//		out = net.forward(a);
		if(abs(out[0] - sol[x]) < 0.25f){
			hit++;
		}
	}
	cout <<"Test: " << (float) hit/ NUM_TEST << endl;
    return 0;
}

int reverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

vector<vector<float>> read_mnist()
{
	ifstream file ("../../train-images.idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number=0;
		int number_of_images=0;
		int n_rows=0;
		int n_cols=0;
		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number= reverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= reverseInt(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);
		vector<vector<float>> out = vector<vector<float>>(number_of_images, vector<float>(n_rows*n_cols));
		for(int i=0;i<number_of_images;++i)
		{
			for(int r=0;r<n_rows;++r)
			{
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					out[i][r*n_rows + c] = temp / 255.f;
				}
			}
		}
		return out;
	}
}

vector<uint8_t> read_label(){
	ifstream file ("../../train-labels.idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_labels = 0;
		file.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char *) &number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverseInt(number_of_labels);
		vector<uint8_t> labels = vector<uint8_t>(number_of_labels);
		for(int i=0; i<number_of_labels; i++) {
			file.read((char *) &labels[i], sizeof(uint8_t));
		}
		return labels;
	}
	exit(1);
}
