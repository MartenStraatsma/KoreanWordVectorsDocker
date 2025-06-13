CXX = c++
CXXFLAGS = -pthread -std=c++2c -march=native
OBJS = args.o koreanargs.o autotune.o matrix.o dictionary.o koreandictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o fasttext.o koreanfasttext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
opt: fasttext

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

koreanargs.o: src/koreanargs.cc src/koreanargs.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/koreanargs.cc

autotune.o: src/autotune.cc src/autotune.h
	$(CXX) $(CXXFLAGS) -c src/autotune.cc

matrix.o: src/matrix.cc src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

koreandictionary.o: src/koreandictionary.cc src/koreandictionary.h src/dictionary.h src/koreanargs.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/koreandictionary.cc

loss.o: src/loss.cc src/loss.h src/matrix.h src/real.h
	$(CXX) $(CXXFLAGS) -c src/loss.cc

productquantizer.o: src/productquantizer.cc src/productquantizer.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

densematrix.o: src/densematrix.cc src/densematrix.h src/utils.h src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/densematrix.cc

quantmatrix.o: src/quantmatrix.cc src/quantmatrix.h src/utils.h src/matrix.h
	$(CXX) $(CXXFLAGS) -c src/quantmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

meter.o: src/meter.cc src/meter.h
	$(CXX) $(CXXFLAGS) -c src/meter.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

koreanfasttext.o: src/koreanfasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/koreanfasttext.cc

fasttext: $(OBJS) src/fasttext.cc src/main.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext

clean:
	rm -rf *.o fasttext