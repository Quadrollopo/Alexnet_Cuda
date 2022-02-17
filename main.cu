#include <iostream>
#include "network.cuh"
#include "CUDA/convolution.cuh"
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>

using namespace std;

#define BATCH_SIZE 64
#define NUM_EPOCHS 1000
#define NUM_TEST 400
#define IMG_LOAD 2000

vector<vector<float>> read_mnist();
vector<uint8_t> read_label();

int main() {

	vector<vector<float>> numbers = read_mnist();
	cout << "numbers loaded" << endl;
	vector<uint8_t> labels = read_label();
	cout << "label loaded" << endl;
	Network net(28, 1,0.7f);
	net.addConvLayer(7, 16, 1, false, reLu)->
	addFullLayer(10, Sigmoid);
	float *out, *sol_dev, *numbers_dev;
	float* sol = new float [10]();
	float* out_h = new float [10]();
	cudaMalloc(&sol_dev, 10 * sizeof(float));
	cudaMalloc(&numbers_dev, numbers[0].size() * sizeof(float));

	random_device r;
	uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, IMG_LOAD - 1);
	double loss;
	for (int j=0; j < NUM_EPOCHS; j++) {
		for (int i = 0; i < BATCH_SIZE; i++) {
			int x = distribution(r);
			sol[labels[x]] = 1;
			cudaMemcpy(numbers_dev, numbers[x].data(), numbers[x].size(), cudaMemcpyHostToDevice);
			out = net.forward(numbers_dev);
			cudaMemcpy(sol_dev, sol, 10, cudaMemcpyHostToDevice);
			net.train(out, sol_dev, numbers_dev);
			if (j % 100 == 0) {
				loss = 0.0;
				cudaMemcpy(out_h, out, 10, cudaMemcpyDeviceToHost);
				for(int z=0; z<10; z++) {
					loss += pow((out_h[z] - sol[z]), 2);
				}
			}
			sol[labels[x]] = 0;
		}
		net.learn();
		if(j % 100 == 0){
			cout << "loss: " << loss / BATCH_SIZE << endl;
			int hit = 0;
			for (int i = 0; i < NUM_TEST; i++) {
				cudaMemcpy(numbers_dev, numbers[i].data(), numbers[i].size(), cudaMemcpyHostToDevice);
				out = net.forward(numbers_dev);
				cudaMemcpy(out_h, out, 10, cudaMemcpyDeviceToHost);

				float max_ix = out_h[0];
				int mx = 0;
				for(int x=1; x<10; x++){
					if(out_h[x] > max_ix){
						max_ix = out_h[x];
						mx = x;
					}
				}
				if(mx == labels[i])
					hit++;
			}
			cout <<"Test: " << (float) hit/ NUM_TEST << endl;
		}
	}


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
		vector<vector<float>> out = vector<vector<float>>(IMG_LOAD, vector<float>(n_rows*n_cols));
		for(int i=0;i<IMG_LOAD;++i)
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
		vector<uint8_t> labels = vector<uint8_t>(IMG_LOAD);
		for(int i=0; i<IMG_LOAD; i++) {
			file.read((char *) &labels[i], sizeof(uint8_t));
		}
		return labels;
	}
	exit(1);
}
