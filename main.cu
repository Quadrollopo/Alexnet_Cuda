#include <iostream>
#include "network.cuh"
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>

using namespace std;

#define BATCH_SIZE 64
#define NUM_EPOCHS 10
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

    int image_size = 2048;
    int kernel_size = 3;
    int pad = 1;
    int stride = 2;
    int image_ch = 3;
    int kernel_ch = 10;



    auto image = new float[image_size*image_size*image_ch];
    auto kernel = new float[kernel_size*kernel_size*kernel_ch*image_ch];
    for(int i=0;i<image_size*image_size*image_ch;i++)
        image[i]=(float)i;
    for(int i=0; i<kernel_ch; i++) {
        for (int j = 0; j < image_ch; j++){
             for (int k = 0; k < kernel_size * kernel_size; k++) {
                kernel[i * kernel_size * kernel_size * image_ch + j * kernel_size * kernel_size + k] = 1;
                        //(float)i * kernel_size * kernel_size * image_ch + j * kernel_size * kernel_size + k + 1;
                //printf("%.1f ",kernel[i * kernel_size * kernel_size * image_ch + j * image_ch + k]);
            }
             //printf("\n");
        }
    }
    float *d_image, *d_kernel, *res, *res2;
    int res_dim = (image_size-kernel_size+2*pad)/stride+1;
    cudaMalloc(&d_image, image_size * image_size * image_ch * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float));
    cudaMalloc(&res, res_dim * res_dim * kernel_ch * sizeof(float));
    cudaMalloc(&res2, image_size * image_size * image_ch * sizeof(float));
    cudaMemset(res2, 0, image_size * image_size * image_ch * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * image_ch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);

    //float* conv_CUDA = convolution(image,kernel,image_size,kernel_size,stride,pad,image_ch,kernel_ch);
    convolution(d_image,d_kernel, res, image_size, kernel_size, stride, pad, image_ch, kernel_ch);
    convolution_prevlayer_backpropagation(res,d_kernel, res2, res_dim, kernel_size, image_size, kernel_ch, image_ch);

    //auto res_CPU = convolution_CPU(image,kernel,kernel_size,image_size,stride,true);
    //delete[] conv_CUDA;
    delete[] image;
    delete[] kernel;
    //delete[] res_CPU;

    cudaFree(d_image);
    cudaFree(d_kernel);

    auto image1 = new float[image_size*image_size];
    auto image2 = new float[image_size*image_size];
    for(int i=0;i<image_size*image_size;i++){
        image1[i]=(float)i+1;
    }
    for(int i=0;i<image_size*image_size;i++){
        image2[i]=(float)i+1;
    }
    float *d_image1, *d_image2, *d_image3;
    cudaMalloc(&d_image1, image_size * image_size * sizeof(float));
    cudaMalloc(&d_image2, image_size * image_size * sizeof(float));
    cudaMalloc(&d_image3, image_size * image_size * sizeof(float));

    cudaMemcpy(d_image1, image1, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image2, image2,  image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_image3, 0,  image_size * image_size * sizeof(float));

    //float* res_CUDA = matrix_mul3(d_image1,d_image2,image_size,image_size,image_size);
    matrix_mul3(d_image1,d_image2, d_image3, image_size,image_size,image_size);

//    delete[] image1;
//    delete[] image2;
//    //cudaFree(res_CUDA);
//    cudaFree(res_CUDA2);
//    cudaFree(d_image1);
//    cudaFree(d_image2);

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
