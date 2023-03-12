
vpath %.h include
vpath %.cpp src
OBJ=main.o single_list.o data_json.o jsoncpp.a
CC=g++

main:$(OBJ)
	$(CC) $(OBJ) -o $@

main.o:main.cpp
single_list.o:single_list.cpp single_list.h
data_json.o:data_json.cpp data_json.h

.PHONY:clean
clean:
	-rm  main *.o
