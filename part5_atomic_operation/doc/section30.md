# histogram operation (통계처리)

- atomic operation 사용하여 histogram 구하기
- another example: histogram
  - host version
    - naive for-loop, byte 단위 처리
    - 실행 시간: 1,684 usec
  - host – 4 byte
    - naive for-loop
    - 속도를 높이기 위해서, 4바이트 단위의 word 단위 처리를 했을때, 얼마나 빨라지는지 알아봄
      - 4 바이트로 한꺼번에 받아와서, 
      - 1 바이트(8비트) 단위로 비트연산으로 끊어서, 받아옴.
    - 실행 시간: 993 usec 
  - CUDA 글로벌 메모리 사용 버전
    - byte 단위(8비트) 처리
    - 2D -> 1D 로 바꾼 다음, 
    - 256000개 쓰래드가 동시에 돌아가면서, 32개의 배열에 업데이트.. 
      - 1개 쓰래드가.. 1개 픽셀을 처리하도록.. 
      - 32개 배열에 업데이트할때 atomic 연산으로 업데이트
    - 실행 시간: 148 usec
  - CUDA 글로벌 메모리, 4 byte 단위 처리 
    - 실행 시간: 136 usec 
  - CUDA shared memory
    - 1바이트 단위 처리, 
    - 32개 ..쓰래드 동시에.. 쓰래드 블럭 단위로.
    - 히스토그램 32개 배열을 쓰래드 블락 마다, 
    - 하나씩 쉐어드 메모리에 올려줘서
    - 이 쉐어드 메모리에 히스토그램을 업데이트 하는 걸로 하면, 
    - 쫌 빨라지지 않을까.
    - 다만, 끝나고 나면, 
      - 이 쉐어드메모리에 있는 히스토그램 배열을 
      - 진짜..글로벌 메모리의 히스토그램 배열로 업데이트해줘야 함
      - 실행 시간: 44 usec 
  - CUDA shared memory, 4 byte 단위 처리
    - 실행 시간: 41 usec


[Return Par5 Atomic Operation](../README.md)   