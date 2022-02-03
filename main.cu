#include <iostream>
#include <vector>
#include "network.h"
#include "matrixMul.cuh"
using namespace std;


int main() {

    Network net(5);
    net.addFullLayer(4);
    net.addFullLayer(3);
    float in[] = {3.0f, 2.0f, 2.5f, 2.0f, 2.0f};
    float *out = net.forward(in);
    for (int i = 0; i < 3; i++)
        cout << out[i] << endl;
    return 0;
}



