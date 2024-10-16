.PHONY: build clean clean-perf

build:
	cmake -B build -S .
	cmake --build build -j $(nproc)

clean-perf:
	rm -f perf.data perf.data.old report*.sqlite report*.nsys-rep

clean: clean-perf
	rm -rf build