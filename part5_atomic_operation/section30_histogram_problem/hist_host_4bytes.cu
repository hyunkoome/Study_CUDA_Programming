#include "./common.cpp"
#include "./image.cpp"

const unsigned image_width = 640;
const unsigned image_height = 400;
const unsigned HIST_SIZE = 32; // histogram levels

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
	uint32_t* ptr = reinterpret_cast<uint32_t*>(grayscale_data);
	for (register unsigned i = 0; i < sizeof(grayscale_data) / 4; ++i) {
		unsigned value = *ptr++;
		vecHist[(value & 0xFF) / 8]++;
		vecHist[((value >> 8) & 0xFF) / 8]++;
		vecHist[((value >> 16) & 0xFF) / 8]++;
		vecHist[((value >> 24) & 0xFF) / 8]++;
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
