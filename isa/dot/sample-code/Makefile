all: bdot-test bdot-test-riscv

run: run-bdot-test run-bdot-test-riscv

run-bdot-test: bdot-test
	./$< 128
	./$< 256
	./$< 512
	./$< 1024

run-bdot-test-riscv: bdot-test-riscv
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl128b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl256b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl512b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvl1024b pk ./$<

bdot-test: main.cc *.h
	g++ -std=c++17 -O2 $< -o $@

bdot-test-riscv: main.cc *.h
	riscv64-unknown-elf-g++ -DIME -march=rv64gcv -O2 $< -o $@

clean:
	rm -f bdot-test bdot-test-riscv
