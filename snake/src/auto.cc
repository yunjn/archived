#include "auto.h"

#include <iostream>
using namespace std;
extern Node food;
extern int g_snake_direction;
extern int g_enter_direction;

void Auto(queue<Node>& snake) {                               //憨憨行为
  if (snake.back().y == food.y && snake.back().x < food.x &&  //反向
      g_snake_direction == 'a')
    g_enter_direction = 'w';
  else if (snake.back().y == food.y && snake.back().x > food.x &&
           g_snake_direction == 'd')
    g_enter_direction = 's';
  else if (snake.back().x == food.x && snake.back().y > food.y &&
           g_snake_direction == 's')
    g_enter_direction = 'a';
  else if (snake.back().x == food.x && snake.back().y < food.y &&
           g_snake_direction == 'w')
    g_enter_direction = 'd';
  else if (snake.back().x <= food.x && snake.back().y <= food.y) {  //左上角
    if (g_snake_direction == 'a' || g_snake_direction == 's') {
      g_enter_direction = 's';
      if (snake.back().y == food.y) g_enter_direction = 'd';
    } else if (g_snake_direction == 'w' || g_snake_direction == 'd') {
      g_enter_direction = 'd';
      if (snake.back().x == food.x) g_enter_direction = 's';
    }
  } else if (snake.back().x >= food.x && snake.back().y < food.y) {  //右上角
    if (g_snake_direction == 'd' || g_snake_direction == 's') {
      g_enter_direction = 's';
      if (snake.back().y == food.y) g_enter_direction = 'a';
    } else if (g_snake_direction == 'w' || g_snake_direction == 'a') {
      g_enter_direction = 'a';
      if (snake.back().x == food.x) g_enter_direction = 's';
    }
  } else if (snake.back().x <= food.x && snake.back().y > food.y) {  //左下角
    if (g_snake_direction == 'a' || g_snake_direction == 'w') {
      g_enter_direction = 'w';
      if (snake.back().y == food.y) g_enter_direction = 'd';
    } else if (g_snake_direction == 's' || g_snake_direction == 'd') {
      g_enter_direction = 'd';
      if (snake.back().x == food.x) g_enter_direction = 'w';
    }
  } else if (snake.back().x >= food.x && snake.back().y >= food.y) {  //右下角
    if (g_snake_direction == 'd' || g_snake_direction == 'w') {
      g_enter_direction = 'w';
      if (snake.back().y == food.y) g_enter_direction = 'a';
    } else if (g_snake_direction == 's' || g_snake_direction == 'a') {
      g_enter_direction = 'a';
      if (snake.back().x == food.x) g_enter_direction = 'w';
    }
  }
}

double dis(Node node_first, Node node_second) {
  return sqrtf(pow(node_first.x - node_second.x, 2) +
               pow(node_first.y - node_second.y, 2));
}

dis_dir min(dis_dir arr[]) {
  auto tmp = arr[0];
  for (int i = 1; i < 3; i++)
    if (tmp.distence > arr[i].distence) tmp = arr[i];
  return tmp;
}

void AutoDis(std::queue<Node>& snake) {
  dis_dir arr[3];
  switch (g_snake_direction) {
    case 'a':
    case 'A':
      arr[0] = {dis({snake.back().x - 1, snake.back().y}, food), 'a'};
      arr[1] = {dis({snake.back().x, snake.back().y + 1}, food), 's'};
      arr[2] = {dis({snake.back().x, snake.back().y - 1}, food), 'w'};
      g_enter_direction = min(arr).direction;
      break;
    case 'd':
    case 'D':
      arr[0] = {dis({snake.back().x + 1, snake.back().y}, food), 'd'};
      arr[1] = {dis({snake.back().x, snake.back().y - 1}, food), 'w'};
      arr[2] = {dis({snake.back().x, snake.back().y + 1}, food), 's'};
      g_enter_direction = min(arr).direction;
      break;
    case 'w':
    case 'W':
      arr[0] = {dis({snake.back().x, snake.back().y - 1}, food), 'w'};
      arr[1] = {dis({snake.back().x - 1, snake.back().y}, food), 'a'};
      arr[2] = {dis({snake.back().x + 1, snake.back().y}, food), 'd'};
      g_enter_direction = min(arr).direction;
      break;
    case 's':
    case 'S':
      arr[0] = {dis({snake.back().x, snake.back().y + 1}, food), 's'};
      arr[1] = {dis({snake.back().x + 1, snake.back().y}, food), 'd'};
      arr[2] = {dis({snake.back().x - 1, snake.back().y}, food), 'a'};
      g_enter_direction = min(arr).direction;
      break;
  }
}