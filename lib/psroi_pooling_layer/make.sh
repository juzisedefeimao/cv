#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0


# add building psroi_pooling layer
#cd psroi_pooling_layer
#nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52
#
#g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
#	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

cd C:\Users\jjj\Desktop\jjj\zlrm\lib\psroi_pooling_layer
nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include"

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc \
	psroi_pooling_op.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -fPIC -lcudart -L $TF_LIB -ltensorflow_framework -L $CUDA_PATH/lib64

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc \
#	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
C:\Users\jjj\AppData\Local\Programs\Python\Python36\Scripts
cl test_op.cpp /LD /I"C:\Users\jjj\AppData\Local\Programs\Python\Python36\Lib\site-packages\tensorflow\include" /I"C:\vs2017\VC\Tools\MSVC\14.14.26428\include" /I"C:\vs2017\SDK\ScopeCppSDK\SDK\include\ucrt" /I"C:\vs2017\SDK\ScopeCppSDK\SDK\include\um" /I"C:\vs2017\SDK\ScopeCppSDK\SDK\include\shared"
cd ..
nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc -I "C:\Users\jjj\AppData\Local\Programs\Python\Python36\Lib\site-packages\tensorflow\include"  -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include"
cl -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc psroi_pooling_op.cu.o -I "C:\Users\jjj\AppData\Local\Programs\Python\Python36\Lib\site-packages\tensorflow\include" -D GOOGLE_CUDA=1 -fPIC -lcudart -L $TF_LIB -ltensorflow_framework -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include"