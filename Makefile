CXX			= CC
CXXFLAGS	= -O3 -Wall -g -std=c++2a

DEPS 	= ilu.h
OBJ 	= ilu.o example_test.o 

all: example_test

example_test: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o *.out *.err $(ALL)

