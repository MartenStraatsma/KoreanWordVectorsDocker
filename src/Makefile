#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -pthread -std=c++17 -march=native
OBJS = args.o autotune.o dictionary.o meter.o fasttext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
opt: fasttext

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

autotune.o: src/autotune.cc src/autotune.h
	$(CXX) $(CXXFLAGS) -c src/autotune.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

meter.o: src/meter.cc src/meter.h
	$(CXX) $(CXXFLAGS) -c src/meter.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc src/main.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext -lfasttext

clean:
	rm -rf *.o fasttext