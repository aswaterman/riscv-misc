all: zvmm-test zvmm-test-riscv

run: run-zvmm-test run-zvmm-test-riscv

run-zvmm-test: zvmm-test
	./$< 128
	./$< 256
	./$< 512
	./$< 1024

run-zvmm-test-riscv: zvmm-test-riscv
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl128b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl256b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl512b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl1024b pk ./$<

zvmm-test: main.cc *.h
	g++ -std=c++17 -O2 $< -o $@

zvmm-test-riscv: main.cc *.h
	riscv64-unknown-elf-g++ -DIME -march=rv64gcv -O2 $< -o $@

clean:
	rm -f zvmm-test zvmm-test-riscv
