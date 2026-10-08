CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
TARGET := build/ibp_parallel
SOURCE := src/ibp_parallel.cpp

.PHONY: all clean smoke reproduce plot

all: $(TARGET)

$(TARGET): $(SOURCE)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

smoke: $(TARGET)
	python3 scripts/reproduce.py --smoke

reproduce: $(TARGET)
	python3 scripts/reproduce.py --threads 1 2 4 --repeats 3

plot:
	python3 scripts/plot_results.py

clean:
	rm -rf build runs
