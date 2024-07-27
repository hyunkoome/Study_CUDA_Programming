// common.cpp

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define _USE_MATH_DEFINES // to use M_PI
#include <math.h>
#if defined(__CUDACC__) 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;

#if ! defined(__CUDACC__) 
#define __host__ 
#define __device__ 
#endif

template <typename TYPE>
__host__ __device__ inline TYPE div_up( TYPE lhs, TYPE rhs ) {
	return (lhs + rhs - 1) / rhs;
	// another implementation: return ((lhs % rhs) == 0) ? (a / b) : (a / b + 1);
}

template <typename TYPE>
__host__ __device__ inline TYPE nextPowerOfTwo( TYPE val ) {
	/*
	     --x;
	     x |= x >> 1;
	     x |= x >> 2;
	     x |= x >> 4;
	     x |= x >> 8;
	     x |= x >> 16;
	     return ++x;
	 */
	int width = sizeof(TYPE) * 8;
	return 1 << (width - __clz(val - 1));
}


#if ! defined(__CUDACC__) 
#undef __host__ 
#undef __device__ 
#endif

// set random value of [0, bound) to dst array
template <typename TYPE>
void setRandomData( TYPE* pDst, long long num, TYPE bound=static_cast<TYPE>(1000)) {
	int32_t bnd = static_cast<int32_t>(bound);
	while (num--) {
		*pDst++ = static_cast<TYPE>(rand() % bnd);
	}
}

// set normalized random value of [0.0, 1.0) to dst array (float or double)
template <typename TYPE>
void setNormalizedRandomData( TYPE* pDst, long long num, TYPE bound=static_cast<TYPE>(1000)) {
	int32_t bnd = static_cast<int32_t>(bound);
	while (num--) {
		*pDst++ = (rand() % bnd) / static_cast<TYPE>(bnd);
	}
}

// get total sum of a 1D array
template <typename TYPE>
TYPE getNaiveSum( const TYPE* pSrc, int num ) {
	register TYPE sum = static_cast<TYPE>(0);
	while (num--) {
		sum += *pSrc++;
	}
	return sum;
}

// get total sum of a 1D array
template <typename TYPE>
TYPE getSum( const TYPE* pSrc, int num ) {
	register TYPE sum = static_cast<TYPE>(0);
	// add 128K elements in a chunk
	const int chunk = 128 * 1024;
	while (num > chunk) {
		register TYPE partial = static_cast<TYPE>(0);
		register int n = chunk;
		while (n--) {
			partial += *pSrc++;
		}
		sum += partial;
		num -= chunk;
	}
	// add remaining elements
	register TYPE partial = static_cast<TYPE>(0);
	while (num--) {
		partial += *pSrc++;
	}
	sum += partial;
	return sum;
}

// get total sum of element-by-element difference
template <typename TYPE>
TYPE getTotalDiff( const TYPE* lhs, const TYPE* rhs, int num ) {
	register TYPE sum = static_cast<TYPE>(0);
	// add 128K elements in a chunk
	const int chunk = 128 * 1024;
	while (num > chunk) {
		register TYPE partial = static_cast<TYPE>(0);
		register int n = chunk;
		while (n--) {
			if (typeid(TYPE) == typeid(unsigned int)) {
				partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
			} else {
				partial += std::abs((*lhs++) - (*rhs++));
			}
		}
		sum += partial;
		num -= chunk;
	}
	// add remaining elements
	register TYPE partial = static_cast<TYPE>(0);
	while (num--) {
		if (typeid(TYPE) == typeid(unsigned int)) {
			partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
		} else {
			partial += std::abs((*lhs++) - (*rhs++));
		}
	}
	sum += partial;
	return sum;
}

unsigned getTotalDiff( const unsigned* lhs, const unsigned* rhs, int num ) {
	register unsigned sum = static_cast<unsigned>(0);
	// add 128K elements in a chunk
	const int chunk = 128 * 1024;
	while (num > chunk) {
		register unsigned partial = static_cast<unsigned>(0);
		register int n = chunk;
		while (n--) {
			partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
		}
		sum += partial;
		num -= chunk;
	}
	// add remaining elements
	register unsigned partial = static_cast<unsigned>(0);
	while (num--) {
		partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
	}
	sum += partial;
	return sum;
}

float getTotalDiff( int num, const float* lhs, const float* rhs ) {
	printf("WARNING: deprecated getTotalDiff() used\n");
	register float sum = 0.0f;
	while (num--) {
		sum += abs(*lhs++ - *rhs++);
	}
	return sum;
}
unsigned getTotalDiff( int num, const unsigned* lhs, const unsigned* rhs ) {
	printf("WARNING: deprecated getTotalDiff() used\n");
	register unsigned sum = 0U;
	while (num--) {
		sum += abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
	}
	return sum;
}

// get RMS (root-mean-square) error
float getRMS( const float* a, const float* b, int size, bool verbose=false ) {
	register float sum = 0.0f;
	register int count = size;
	float max_err = 0.0f;
	int max_pos = 0;
	while (count--) {
		float err = (*a++) - (*b++);
		sum += err * err;
		if (err > max_err) {
			max_err = err;
			max_pos = (size - (count + 1));
		}
	}
	if (verbose) {
		printf("getRMS: max_err = %f, pos = %d\n", max_err, max_pos);
	}
	return sqrtf( sum / size );
}

template <typename TYPE>
void printVec( const char* name, const TYPE* vec, int num ) {
#if 0
	printf("%s = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n", name,
	       vec[0], vec[1], vec[2], vec[3], vec[num - 4], vec[num - 3], vec[num - 2], vec[num - 1]);
#endif
	std::streamsize ss = std::cout.precision();
	std::cout.precision(5);
	std::cout << name << "=[";
	std::cout << fixed << showpoint << std::setw(8) << vec[0] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[1] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[2] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[3] << " ... ";
	std::cout << fixed << showpoint << std::setw(8) << vec[num - 4] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[num - 3] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[num - 2] << " ";
	std::cout << fixed << showpoint << std::setw(8) << vec[num - 1] << "]" << std::endl;
	std::cout.precision(ss);
}


// print all elements in the vector
template <typename TYPE>
void printVecAll( const char* name, const TYPE* vec, int num ) {
	std::cout << std::setw(8) << name << "=[";
	while (num--) {
		std::cout << std::setw(8) << *vec++ << " ";
	}
	std::cout << "]" << std::endl;
}

// print a matrix 
void printMat( const char* name, float* mat, int nrow, int ncol ) {
	int c = ncol;
#define M(row,col) mat[(row)*ncol+(col)]
	printf("%s=[", name);
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(0, 0), M(0, 1), M(0, 2), M(0, c - 3), M(0, c - 2), M(0, c - 1));
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(1, 0), M(1, 1), M(1, 2), M(1, c - 3), M(1, c - 2), M(1, c - 1));
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(2, 0), M(2, 1), M(2, 2), M(2, c - 3), M(2, c - 2), M(2, c - 1));
	printf("\t........ ........ ........ ... ........ ........ ........\n");
	int r = nrow - 3;
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
	r = nrow - 2;
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
	r = nrow - 1;
	printf("\t%8f %8f %8f ... %8f %8f %8f ]\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
#undef M
}

// print a 3D array
void print3D( const char* name, float* ptr, int dimz, int dimy, int dimx ) {
	char buf[20];
	sprintf(buf, "%s[0]", name);
	printMat( buf, ptr, dimy, dimx );
	float* last = ptr + (dimz - 1) * (dimy * dimx);
	sprintf(buf, "%s[%d]", name, dimz - 1);
	printMat( buf, last, dimy, dimx );
}

// CUDA error check macro
#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define	CUDA_CHECK(x)	do { \
		(x); \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			printf("CUDA FAILURE \"%s\" at %s:%d\n", \
			         cudaGetErrorString(err), __FILE__, __LINE__); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)
#endif

// CAUTION: we check CUDA error even in release mode
// #if defined(NDEBUG)
// #define CUDA_CHECK_ERROR()  0
// #else
#define CUDA_CHECK_ERROR()  do { \
        cudaError_t e = cudaGetLastError(); \
        if (cudaSuccess != e) { \
            printf("cuda failure \"%s\" at %s:%d\n", \
                   cudaGetErrorString(e), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
// #endif


chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

#define ELAPSED_TIME_BEGIN(N)	do { \
		__time_begin[(N)] = chrono::system_clock::now(); \
		printf("elapsed wall-clock time[%d] started\n", (N)); \
		fflush(stdout); \
	} while (0)

#define ELAPSED_TIME_END(N)	do { \
		chrono::system_clock::time_point time_end = chrono::system_clock::now(); \
		chrono::microseconds elapsed_msec = chrono::duration_cast<chrono::microseconds>(time_end - __time_begin[(N)]); \
		printf("elapsed wall-clock time[%d] = %ld usec\n", (N), (long)elapsed_msec.count()); \
		fflush(stdout); \
	} while (0)

// print kernel configuration

#define CUDA_PRINT_CONFIG(dimx)	do { \
		printf("prob size = %d\n", dimx); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

#define CUDA_PRINT_CONFIG_2D(dimx,dimy)	do { \
		printf("prob size = %d * %d\n", dimx, dimy); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

#define CUDA_PRINT_CONFIG_3D(dimx,dimy,dimz)	do { \
		printf("prob size = %d * %d * %d\n", dimx, dimy, dimz); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

// argument processing

template <typename TYPE>
TYPE procArg( const char* progname, const char* str, TYPE lbound = -1, TYPE ubound = -1) {
	char* pEnd = nullptr;
	TYPE value = 0;
	if (typeid(TYPE) == typeid(float) && typeid(TYPE) == typeid(double)) {
		value = strtof( str, &pEnd );
	} else {
		value = strtol( str, &pEnd, 10 );
	}
	// extra suffix processing
	if (typeid(TYPE) != typeid(float) && typeid(TYPE) != typeid(double)) {
		if (pEnd != nullptr && *pEnd != '\0') {
			switch (*pEnd) {
			case 'k':
			case 'K':
				value *= 1024;
				break;
			case 'm':
			case 'M':
				value *= (1024 * 1024);
				break;
			case 'g':
			case 'G':
				value *= (1024 * 1024 * 1024);
				break;
			default:
				printf("%s: ERROR: illegal parameter '%s'\n", progname, str);
				exit(EXIT_FAILURE); // EINVAL: invalid argument
				break;
			}
		}
	}
	// check for bounds
	if (lbound != -1 && value < lbound) {
		printf("%s: ERROR: invalid value '%s'\n", progname, str);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	if (ubound != -1 && value > ubound) {
		printf("%s: ERROR: invalid value '%s'\n", progname, str);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	// done
	return value;
}


// ASSERT from <assert.h>

#if ! defined(_MSC_VER)
// linux case
#define ASSERT(expr) \
	(static_cast <bool> (expr)                     \
	 ? void (0)                            \
	 : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
#else
// Visual Studio case
#define ASSERT(expression) (void)(                                                       \
        (!!(expression)) ||                                                              \
        (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
                                 )
#endif

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
