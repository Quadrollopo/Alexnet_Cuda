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
#define IMG_LOAD 60000
#define DECREASE_STEP 600

vector<vector<float>> read_mnist();
vector<uint8_t> read_label();

int main() {
	vector<vector<float>> numbers = read_mnist();
	cout << "numbers loaded" << endl;
	vector<uint8_t> labels = read_label();
	cout << "label loaded" << endl;

	Network net(28, 1,5e-2);
	net.addConvLayer(7, 16, 3, false, reLu)->
	addPoolLayer(2, 2)->
	addConvLayer(3, 32, 1, true, reLu)->
//	addConvLayer(3, 32, 1, true, reLu)->
	addPoolLayer(2, 2)->
	addFullLayer(256, reLu)->
	addFullLayer(256, reLu)->
	addFullLayer(10, softmax);

	random_device r;
	uniform_int_distribution<int> generator = uniform_int_distribution<int>(0 , IMG_LOAD/NUM_TEST - 1);
	float *out, *sol_dev, *numbers_dev;
	float* sol = new float [10]();
	float* out_h = new float [10]();
	cudaMalloc(&sol_dev, 10 * sizeof(float));
	cudaMalloc(&numbers_dev, numbers[0].size() * sizeof(float));
	vector<int> test_index = vector<int>(NUM_TEST);
	for(int i=0; i<NUM_TEST; i++){
		test_index[i] = generator(r) + i * (IMG_LOAD/NUM_TEST);
	}

	float loss;
	int x = 0;
	ofstream hist_file("./history.txt");
	ofstream val_file("./val.txt");
	for (int j=0; j < NUM_EPOCHS; j++) {
		loss = 0.0;
		for (int i = 0; i < BATCH_SIZE; i++, x++) {
			x = x % IMG_LOAD;
			sol[labels[x]] = 1;
			cudaMemcpy(numbers_dev, numbers[x].data(), numbers[x].size(), cudaMemcpyHostToDevice);
//			print_CUDA(numbers_dev, numbers[x].size());
			out = net.forward(numbers_dev);
			cudaMemcpy(sol_dev, sol, 10 * sizeof(float), cudaMemcpyHostToDevice);
			net.train(out, sol_dev, numbers_dev);
			cudaMemcpy(out_h, out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
			for(int z=0; z < 10; z++)
				loss += -((sol[z] * log(out_h[z] + 1e-10f)) + (1-sol[z]) * log(1-out_h[z] + 1e-10f));

			sol[labels[x]] = 0;
		}
		net.learn();
		loss /= 10 * BATCH_SIZE;
		cout << "loss: " << loss << endl;
		hist_file << loss << " ";
		if(j % 10 == 9) {
			int hit = 0;
			for (int i = 0; i < NUM_TEST; i++) {
				int z = test_index[i];
				cudaMemcpy(numbers_dev, numbers[z].data(), numbers[z].size(), cudaMemcpyHostToDevice);
				out = net.forward(numbers_dev);
				cudaMemcpy(out_h, out, 10 * sizeof(float), cudaMemcpyDeviceToHost);

				float max_ix = out_h[0];
				int mx = 0;
				for (int m = 1; m < 10; m++) {
					if (out_h[m] > max_ix) {
						max_ix = out_h[m];
						mx = m;
					}
				}
				if (mx == labels[z])
					hit++;
			}
			cout <<"Test: " << (float) hit/ NUM_TEST << endl;
			val_file << (float) hit/ NUM_TEST << " ";
		}
		if(j % DECREASE_STEP == DECREASE_STEP - 1)
			net.decreaseLR();

	}
	hist_file.close();
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
	ifstream file ("../train-images.idx3-ubyte", ios::binary);
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
		vector<float> val = vector<float>(n_rows*n_cols);
		for(int i=0;i<IMG_LOAD;++i)
		{
			for(int r=0;r<n_rows;++r)
			{
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					val[r*n_rows + c] = temp / 255.f;
				}
			}
			out[i] = val;
		}
		return out;
	}
	exit(1);
}

vector<uint8_t> read_label(){
	ifstream file ("../train-labels.idx1-ubyte", ios::binary);
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
