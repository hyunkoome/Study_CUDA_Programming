#include <stdio.h>

int main(int argc, char* argv[], char* envp[]) {
	printf("argc = %d\n", argc);
	for (int i = 0; i < argc; ++i) {
		printf("argv[%d] = \"%s\"\n", i, argv[i]);
	}
	for (int i = 0; envp[i] != nullptr; ++i) {
		printf("envp[%d] = \"%s\"\n", i, envp[i]);
	}
    // envp[k] = (char*)0 // 끝을 의미 (== nullptr)
	// done
	return 0;
}
