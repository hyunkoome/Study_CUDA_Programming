/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */

#include <stdio.h>
#include <time.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

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
	system_clock::time_point chrono_begin = system_clock::now();
	// work
	bigJob();
	// work done
	system_clock::time_point chrono_end = system_clock::now();
	// calculation
	microseconds chrono_elapsed_usec = duration_cast<microseconds>(chrono_end - chrono_begin);
	printf("elapsed time = %ld usec\n", (long)chrono_elapsed_usec.count());
	// done
	return 0;
}

// usec = micro second = 1/1,000,000 sec
