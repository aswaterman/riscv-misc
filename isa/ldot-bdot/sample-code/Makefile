all: bdota-test bdota-test-riscv

run: run-bdota-test run-bdota-test-riscv

run-bdota-test: bdota-test
	./$< 128
	./$< 256
	./$< 512
	./$< 1024

run-bdota-test-riscv: bdota-test-riscv
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvqwbdota8i_zvqwbdota16i_zvfwbdota16bf_zvfbdota32f_zvl128b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvqwbdota8i_zvqwbdota16i_zvfwbdota16bf_zvfbdota32f_zvl256b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvqwbdota8i_zvqwbdota16i_zvfwbdota16bf_zvfbdota32f_zvl512b pk ./$<
	spike --isa=rv64gcv_zicsr_zicntr_zifencei_zvfh_zvfbfmin_zvqwbdota8i_zvqwbdota16i_zvfwbdota16bf_zvfbdota32f_zvl1024b pk ./$<

bdota-test: main.cc *.h
	g++ -std=c++17 -O2 $< -o $@

bdota-test-riscv: main.cc *.h
	riscv64-unknown-elf-g++ -march=rv64gcv -O2 $< -o $@

clean:
	rm -f bdota-test bdota-test-riscv
