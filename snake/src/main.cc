#include "auto.h"
#include "features.h"
int main() {
  //隐藏光标
  HANDLE h_out = GetStdHandle(STD_OUTPUT_HANDLE);
  CONSOLE_CURSOR_INFO cursor_info;
  GetConsoleCursorInfo(h_out, &cursor_info);
  cursor_info.bVisible = false;
  SetConsoleCursorInfo(h_out, &cursor_info);

  srand((unsigned int)time(NULL));
  std::queue<Node> snake;
  extern int g_num_of_food_eaten;
  int speed;
  do {
    Welcome(speed);
    GameInit(snake);
    while (true) {
      Sleep(1000 / speed);
      CreateFood();
      MoveSnake(snake);
      // Auto(snake);
      AutoDis(snake);
      SnakeGrowUp(snake);
      if (!IsSnakeDie(snake) || g_num_of_food_eaten >= 2000) break;
    }
    if (g_num_of_food_eaten >= 2000)
      MessageBox(NULL, TEXT("YOU WIN !!!"), TEXT("Greedy Snake"),
                 MB_OK | MB_ICONERROR);
    Clean(snake);
  } while (GameOver(speed));

  return 0;
}