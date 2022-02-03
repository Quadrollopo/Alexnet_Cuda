#include <iostream>
#include "network.h"
using namespace std;


int main() {

    Network net(5);
    net.addFullLayer(4);
    net.addFullLayer(3);
    float in[] = {3.0f, 2.0f, 2.5f, 2.0f, 2.0f};
    float *out = net.forward(in);
    for (int i = 0; i < 3; i++)
        cout << out[i] << endl; //46
    float a[] = {0.0f, 1.0f, 0.0f};
    net.learn(out, a);
    return 0;
}
