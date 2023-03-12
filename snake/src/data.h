#ifndef DATA_H
#define DATA_H
#pragma warning(disable : 4996)
#define _CRT_SECURE_NO_WARNINGS
#define MAP_MODE "■"
#define SNAKE_MODE "■"
#define FOOD_MODE "●"
#define MAP_WIDTH 70
#define MAP_HEIGHT 30
#define LIGHT_GREEN "\033[1;32m"
#define RED "\033[0;32;31m"
#define WHITE "\033[1;37m"
#define LIGHT_BLUE "\033[1;34m"

typedef struct Node {
  int x;
  int y;
} Node;

typedef struct DisDir {
  double distence;
  char direction;
} dis_dir;

#endif