#include "./common.cpp"
#include "./image.cpp"

const unsigned image_width = 640;
const unsigned image_height = 400;
const unsigned HIST_SIZE = 32; // histogram levels

// 640x400 image = 256,000 pixels
// each pixel: 256 level values: 0 (0x00) ~ 255 (0xff), 16진수

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	default:
		printf("usage: %s\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	unsigned* vecHist = nullptr;
	try {
		vecHist = new unsigned[HIST_SIZE];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set data to be zero
	memset(vecHist, 0, HIST_SIZE * sizeof(unsigned));
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	unsigned char* ptr = grayscale_data;
	for (register unsigned i = 0; i < sizeof(grayscale_data); ++i) {
		unsigned ind = (*ptr++) / 8; // 256 levels / 8 --> 32 levels
		assert(ind >= 0 && ind < 32);
		vecHist[ind]++;
	}
	ELAPSED_TIME_END(0);
	// check the result
	printf("image pixels = %zu\n", sizeof(grayscale_data));
	printf("histogram levels = %u\n", HIST_SIZE);
	unsigned sum = 0;
	for (register unsigned i = 0; i < HIST_SIZE; ++i) {
		printf("hist[%2d] = %8u\n", i, vecHist[i]);
		sum += vecHist[i];
	}
	printf("sum = %u\n", sum);
	// cleaning
	delete[] vecHist;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
