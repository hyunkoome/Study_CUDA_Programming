#include "./common.cpp"

// dummy big job
void bigJob(void) {
	int count = 0;
	for (int i = 0; i < 10000; ++i) {
		for (int j = 0; j < 10000; ++j) {
			count++;
		}
	}
	printf("we got %d counts.\n", count);
}

int main(void) {
	ELAPSED_TIME_BEGIN(0);
	// work
	bigJob();
	// work done
	ELAPSED_TIME_END(0);
	// done
	return 0;
}