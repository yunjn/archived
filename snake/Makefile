CXX = g++
vpath %.h src
vpath %.cc src

OBJS = main.o features.o auto.o
OBJS_DIR = build

FLAG_EGE = -lgraphics64 -luuid -lmsimg32 -lgdi32 -limm32 -lole32 -loleaut32 -lwinmm -lgdiplus
FLAG_GBK = -fexec-charset=GBK

snake:$(OBJS)
	$(CXX) build/main.o build/features.o build/auto.o $(FLAG_GBK) -o $@

%.o:%.cc
	$(CXX) -c $< $(FLAG_GBK) -o $(OBJS_DIR)/$@

.PHONY:clean
clean:
	-del snake.exe
	-del build\*.o