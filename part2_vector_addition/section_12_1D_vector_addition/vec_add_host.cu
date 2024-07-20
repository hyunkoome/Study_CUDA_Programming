#include "./common.cpp"

// 100만개 난수 발생하여 데이터 입력
// 0.0 <= x < 1 , 즉, [0.000, 1.000) 의 float 형식 난수 생성
// num = (rand() % 1000) / 1000.0F

const unsigned SIZE = 1024 * 1024; // 100만개, 1M elements

// set random value of [0.000, 1.000) to dst array
void setRandomData( float* dst, int size ) {
	while (size--) {
		*dst++ = (rand() % 1000) / 1000.0F;
	}
}

// get total sum of dst array
float getSum( float* dst, int size ) {
	register float sum = 0.0F;
	while (size--) {
		sum += *dst++;
	}
	return sum;
}

int main( void ) {
	// host-side data
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];
	// set random data
    // seed 에 0을 넣으면, random number generator 가 0으로 reset 됨
    // 다음에 이 프로그램을 실행했을때 같은 데이터를 얻기 위해서, 동일한 seed 값을 줌
	srand( 0 );
	setRandomData( vecA, SIZE );
	setRandomData( vecB, SIZE );
	// kernel execution
	chrono::system_clock::time_point time_begin = chrono::system_clock::now(); // chrono 사용해서, 시작 시간 구함
	for (register int i = 0; i < SIZE; ++i) {
		vecC[i] = vecA[i] + vecB[i];
	}
	chrono::system_clock::time_point time_end = chrono::system_clock::now(); // chrono 사용해서, 끝 시간 구함
	chrono::microseconds time_elapsed_msec = chrono::duration_cast<chrono::microseconds>(time_end - time_begin);
	printf("elapsed wall-clock time = %ld usec\n", (long)time_elapsed_msec.count()); // micro second 단위

    // check the result, 검산을 위해, sum을 각각 구해서, sumC - (sumA + sumB) = 0이 되어야 함
    // 이론상은 sumC - (sumA + sumB) = 0.0 나와햐 하는데,
    // 실제로 해보면 float 를 100만개 더하다 보면, 오차가 생기므로, .. 0.0이 아님
    // 그래서, diff(sumC, sumA+sumB) / SIZE 는 전체 오차에서 SIZE로 나눈 값이어서, 개당 오차 임
    // 즉, diff(sumC, sumA+sumB) / SIZE 가 거의 0에 접근하면, 전체적으로 계산이 맞다고 볼수 있음
	float sumA = getSum( vecA, SIZE );
	float sumB = getSum( vecB, SIZE );
	float sumC = getSum( vecC, SIZE );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);

    // 다 찍는 거 보다는, 처음 4개, 뒤에 4개만 출력
	printf("vecA = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecA[0], vecA[1], vecA[2], vecA[3], vecA[SIZE - 4], vecA[SIZE - 3], vecA[SIZE - 2], vecA[SIZE - 1]);
	printf("vecB = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecB[0], vecB[1], vecB[2], vecB[3], vecB[SIZE - 4], vecB[SIZE - 3], vecB[SIZE - 2], vecB[SIZE - 1]);
	printf("vecC = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecC[0], vecC[1], vecC[2], vecC[3], vecC[SIZE - 4], vecC[SIZE - 3], vecC[SIZE - 2], vecC[SIZE - 1]);
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	// done
	return 0;
}
