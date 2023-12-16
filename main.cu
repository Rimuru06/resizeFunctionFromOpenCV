#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int32_t* deviceDataResized;
int32_t* deviceData;
int32_t* hostOriginalImage;
int32_t* hostResizedImage;

__global__ void resizeCV(int32_t* originalImage, int32_t* resizedImage, int w, int h, int w2, int h2){
	__shared__ int32_t tile[1024];
	const float x_ratio = ((float)(w - 1)) / w2;
	const float y_ratio = ((float)(h - 1)) / h2;
	unsigned int threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread;
	unsigned int shift = 0;
	while((threadId < w2*h2 && shift<elemsPerThread)){
		const int32_t i = threadId / w2;
		const int32_t j = threadId - (i*w2);
		
		const int32_t x = (int)(x_ratio * j);
		const int32_t y = (int)(y_ratio * i);
		const float x_diff = (x_ratio * j) - x;
		const float y_diff = (y_ratio * i) - y;
		const int32_t index = (y*w + x);
		const int32_t a = originalImage[index];
		const int32_t b = originalImage[index + 1];
		const int32_t c = originalImage[index + w];
		const int32_t d = originalImage[index + w + 1];
		const float blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
			(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

		const float green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

		const float red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

		tile[threadIdx.x] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);

		threadId++;
		shift++;
	}
	
	__syncthreads();
	threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread;
	resizedImage[threadId] = tile[threadIdx.x];
}

int32_t* resizeBilinearGPU(int w, int h, int w2, int h2){
    cudaMemcpy(deviceData, hostOriginalImage, w*h * sizeof(int32_t), cudaMemcpyHostToDevice);
	dim3 threads = dim3(threadNum, 1,1);
	dim3 blocks = dim3(w2*h2/ threadNum*elemsPerThread, 1,1);
	resizeCV<<<blocks, threads>>>(deviceData, deviceDataResized, w, h, w2, h2);
    cudaDeviceSynchronize();
	cudaMemcpy(hostResizedImage, deviceDataResized, length * sizeof(int32_t), cudaMemcpyDeviceToHost);

	return hostResizedImage;
}


int main(int argc, char **argv){
	cv::Mat image;
	cv::Mat image_resized_gpu;
	int32_t *argb_res_gpu = NULL;
    int maxResolutionX = 4096;
    int maxResolutionY = 4096;
	cv::Size newSz(1920/2, 1080/2);

	const char fname[] = ""; //Caminho para a imagem
	image = cv::imread(fname, 1);

    int32_t *argb = new int32_t[srcImage.cols*srcImage.rows];
	int offset = 0;

	for (int i = 0; i<srcImage.cols*srcImage.rows * 3; i += 3)
	{
		int32_t blue = image.data[i];
		int32_t green = image.data[i + 1];
		int32_t red = image.data[i + 2];
		argb[offset++] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);
	}


    cudaMalloc((void**)&deviceDataResized, maxResolutionX*maxResolutionY * sizeof(int32_t));
	cudaMalloc((void**)&deviceData, maxResolutionX*maxResolutionY * sizeof(int32_t));=
	argb_res_gpu = resizeBilinearGPU(image.cols, image.rows, newSz.width, newSz.height);
	for (int i = 0; i < RESIZE_CALLS_NUM; i++){
		argb_res_gpu = resizeBilinearGPU(image.cols, image.rows, newSz.width, newSz.height);
	}
    

	image_resized_gpu = cv::Mat(newSz, CV_8UC3);
	cvtInt322Mat(argb_res_gpu, image_resized_gpu);
	cv::imshow("Original", image);
	cv::imshow("Resized_GPU", image_resized_gpu);
	cv::waitKey(0);

	
	cudaFree(deviceDataResized);
	cudaFree(deviceData);

	return 0;
}