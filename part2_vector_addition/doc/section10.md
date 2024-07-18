# Vector addition

### Elapsed Time (소요된 시간 측정)

***modern C++ 의미***
- C++11 and later
- = C++0x
- GNU C++ compiler requires the command line parameter
    - `–std=c++11` to compile C++11 code.
    - `–std=c++0x` also works.
- Microsoft `Visual Studio 2015 (v14) and later` have complete support for
  `C++11` features.
- GNU Compiler can support `–std=c++11`, `–std=c++14`, `–std=c++17`, ... 


***Wall-clock time vs CPU time***
- Wall-clock time (= elapsed real time)
  - 컴퓨터 프로그램이 실행되면서 실제로 흘거간 시간
  - 이론상 벽시계(wall-clock)fh cmrwjd goeh rkxdms rufrhk
- CPU time (CUDA 에서는 GPU time)
  - 컴퓨터 프로그램이 실행 중에 CPU를 사용한 시간
    - system time, 운영체제 kernel time: 운영체제 OS 영역에서 사용한 시간
    - user time, user CPU time: 내 프로그램이 (사용자 user 영역에서 사용한) 돌아가는 시간
- 일반적으로 CPU time < wall-clock time
  - 다만, 병렬 프로그램에서는 CPU time > wall-clock time 가능 
    - 이론 상 가능할수 있지만, 실제로는 거의 일어나지 않음 

***시간 측정 방법***
- C++의 Chrono(크로노) 기능 사용하기
  - chronograph : 스톱워치 기능이 있는 (클래식) 시계
- C++ 11 standard
  - We need a `system-independent` time measuring method...
  - with more high precision
- `#include <chrono>`
  - `using namespace std::chrono` 필수로 적어야 함
  - provide the `nano-second` precision (10^-9)
  - for wall-clock time !
  - (1 / 1,000) sec: `typedef duration<long long, milli> milliseconds;`  
  - (1 / 1,000,000) sec: `typedef duration<long long, micro> microseconds;` // 주로 사용   
  - (1 / 1,000,000,000) sec: `typedef duration<long long, nano> nanoseconds;`
- clock( ) function
  - `#include <time.h>`
  - `clock_t clock( void );`
    - returns an approximation of `processor time (CPU/GPU time)` used by the program
    - to get the number of seconds used, divide by `CLOCKS_PER_SEC`. 시간당 클
  - `float clock_sec = (float)clock() / CLOCKS_PER_SEC;`
  - `long clock_usec = (long)(clock()) * 1000000 / CLOCKS_PER_SEC;`
- sleep( ) function
  - pause the thread
    - making the calling thread to sleep for the specified time periods
    - `wall-clock time 기준` (CPU time은 최소로 사용)
  - `Unix/Linux`
    - `#include <unistd.h>` 유닉스 스탠다드.h
    - 초 단위: `unsinged int sleep( unsigned int seconds );`
    - micro-seconds (1/ 1,000,000) 단위: `int usleep( useconds_t usec );`
  - Windows
    - `#include <windows.h>`
    - milli-seconds (1 / 1,000) 단위: `void Sleep( DWORD dwMilliseconds );`


***Variable Arguments***
- C/C++ main( ) 함수에서 argument 받는 방법
- `int main( int argc, char* argv[], char* envp[] ) { … }`
- 명령어 창: `./a.exe alpha bravo charlie`
  - 시스템에서 자동 생성
  - argc = 4;   // argc = argument count
  - argv[0] = “./a.exe”;  // argv = argument vector (1D array)
  - argv[1] = “alpha”;
  - argv[2] = “bravo”;
  - argv[3] = “charlie”;
  - envp[…] = 환경 변수 environment variable 대입;
    - `envp[k] = (char*)0` // `끝` 을 의미 (== `nullptr`)

***환경 변수 Environment Variables***
- environment variables
  - shell 에서 관리하는 특별한 global
  - 일종의 global 변수 -> 모든 prcess 가 자동으로 상속 받음
- PATH : an environment variable
  - 실행 파일을 검색할 directory 들을 저장, 실행 파일이 어디에 있냐?
- HOME : your home directory (or folder) 홈디렉토리가 무엇이냐?
- USER : your user login nam 유저 계정이름이 뭐냐?


[Return Par2 Vector Addition](../README.md)  