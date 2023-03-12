#include "features.h"

using namespace std;
int g_enter_direction = 'a';
int g_snake_direction = 'a';
int g_is_produce_food = 1;
int g_num_of_food_eaten = 0;
int g_is_del_tail = 1;
COORD g_position;
HANDLE g_out;
Node food, snake_node;

char g_game_map[30][107] = {
    "■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■                                                                  ■\n",
    "■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n",
};

void GameMap() {
  system("color 09");
  system("mode con cols=70 lines=31");
  for (int i = 0; i < 30; i++) printf(g_game_map[i]);
}

void MoveCursor(int xpoint, int ypoint) {
  g_position.X = xpoint * 2;
  g_position.Y = ypoint;
  g_out = GetStdHandle(STD_OUTPUT_HANDLE);
  SetConsoleCursorPosition(g_out, g_position);
}

bool GameOver(int speed) {
  system("cls");
  char sel = '0';
  GameMap();
  MoveCursor((MAP_WIDTH - 42) / 2, MAP_HEIGHT / 3 - 1);
  cout << RED "Game    Over!" WHITE;

  MoveCursor((MAP_WIDTH - 49) / 2, MAP_HEIGHT / 3 + 1);
  cout << LIGHT_GREEN " name:    \t    \t    " WHITE "NULL";

  MoveCursor((MAP_WIDTH - 49) / 2, MAP_HEIGHT / 3 + 3);
  cout << LIGHT_GREEN " rank:    \t    \t    " WHITE "NULL";

  MoveCursor((MAP_WIDTH - 49) / 2, MAP_HEIGHT / 3 + 5);

  cout << LIGHT_GREEN " score:    \t    \t    " WHITE << fixed
       << setprecision(2)
       << (double)g_num_of_food_eaten * log((speed - 4.0) + 1.0);

  MoveCursor((MAP_WIDTH - 54) / 2, MAP_HEIGHT - 12);
  cout << RED " <0.exit>     \t\t    <1.again>" WHITE;

  do {
    sel = _getch();
    if (sel == '1') {
      g_num_of_food_eaten = 0;
      return true;
    } else if (sel == '0')
      exit(0);
  } while (1);
  return false;
}

void SpeedSetting(int& speed) {
  while (true) {
    system("cls");
    cout << "\n\n\n\n\n\n\n   \t   \t   speed<1-100>: ";
    cin >> speed;
    if (speed >= 1 && speed <= 100) {
      speed += 5;
      return;
    } else {
      MessageBox(NULL, TEXT("Please enter the correct option!"),
                 TEXT("Greedy Snake"), MB_OK | MB_ICONERROR);
    }
  }
}

void Welcome(int& speed) {
  system("mode con cols=60 lines=18");
  char sel;
  cout << "\n\n\n\t********************************************" << endl;
  cout << "\t*                                          *" << endl;
  cout << "\t*              Greedy  Snake               *" << endl;
  cout << "\t*                                          *" << endl;
  cout << "\t*        1.simple          2.medium        *" << endl;
  cout << "\t*                                          *" << endl;
  cout << "\t*        3.difficult       4.setting       *" << endl;
  cout << "\t*                                          *" << endl;
  cout << "\t*                                          *" << endl;
  cout << "\t* V0.5                           By samuel *" << endl;
  cout << "\t********************************************" << endl;
  while (1) {
    sel = _getch();
    if (sel >= '0' && sel <= '4') {
      switch (sel) {
        case '1':
          speed = 8;
          return;
        case '2':
          speed = 10;
          return;
        case '3':
          speed = 25;
          return;
        case '4':;
          SpeedSetting(speed);
          return;
      }
    } else {
      MessageBox(NULL, TEXT("Please enter the correct option!"),
                 TEXT("Greedy Snake"), MB_OK | MB_ICONERROR);
    }
  }
}

void CreateFood() {
  if (g_is_produce_food) {
    food.x = rand() % 30 + 2;
    food.y = rand() % 26 + 2;
    MoveCursor(food.x, food.y);
    printf(RED FOOD_MODE WHITE);
    g_is_produce_food = 0;
  }
}

void GameInit(queue<Node>& snake) {
  GameMap();
  g_enter_direction = 'a';
  g_snake_direction = 'a';
  g_is_produce_food = 1;
  g_num_of_food_eaten = 0;
  g_is_del_tail = 1;

  int xpoint = rand() % 21 + 6;
  int ypoint = rand() % 16 + 6;

  snake.push({xpoint + 2, ypoint});
  snake.push({xpoint + 1, ypoint});
  snake.push({xpoint, ypoint});

  for (int i = 0; i < 3; i++) {
    MoveCursor(xpoint + i, ypoint);
    cout << WHITE SNAKE_MODE;
  }
}

void DelTail(queue<Node>& snake) {
  if (g_enter_direction != 'w' && g_enter_direction != 'W' &&
      g_enter_direction != 's' && g_enter_direction != 'S' &&
      g_enter_direction != 'a' && g_enter_direction != 'A' &&
      g_enter_direction != 'd' && g_enter_direction != 'D')
    return;
  MoveCursor(snake.front().x, snake.front().y);
  printf("  ");
  snake.pop();
}

void MoveSnake(queue<Node>& snake) {
  if (_kbhit()) {
    fflush(stdin);
    g_enter_direction = _getch();
  }
  switch (g_enter_direction) {
    case 'W':
    case 'w':
      if (g_snake_direction != 's' && g_snake_direction != 'S') {
        g_snake_direction = g_enter_direction;
        snake.push({snake.back().x, snake.back().y - 1});
      } else
        snake.push({snake.back().x, snake.back().y + 1});
      break;
    case 'S':
    case 's':
      if (g_snake_direction != 'w' && g_snake_direction != 'W') {
        g_snake_direction = g_enter_direction;
        snake.push({snake.back().x, snake.back().y + 1});
      } else
        snake.push({snake.back().x, snake.back().y - 1});
      break;
    case 'D':
    case 'd':
      if (g_snake_direction != 'a' && g_snake_direction != 'A') {
        g_snake_direction = g_enter_direction;
        snake.push({snake.back().x + 1, snake.back().y});
      } else
        snake.push({snake.back().x - 1, snake.back().y});
      break;
    case 'A':
    case 'a':
      if (g_snake_direction != 'd' && g_snake_direction != 'D') {
        g_snake_direction = g_enter_direction;
        snake.push({snake.back().x - 1, snake.back().y});
      } else
        snake.push({snake.back().x + 1, snake.back().y});
      break;
  }
  MoveCursor(snake.back().x, snake.back().y);
  printf(SNAKE_MODE);
  if (g_is_del_tail) DelTail(snake);
  g_is_del_tail = 1;
}

void SnakeGrowUp(queue<Node>& snake) {
  if (food.x == snake.back().x && food.y == snake.back().y) {
    g_num_of_food_eaten++;
    g_is_produce_food = 1;
    g_is_del_tail = 0;
  }
}

bool IsSnakeDie(queue<Node>& snake) {
  // head
  if (snake.back().x == 0 || snake.back().x * 2 == MAP_WIDTH - 2 ||
      snake.back().y == 0 || snake.back().y == MAP_HEIGHT - 1) {
    return false;
  }
  return true;
}

void Clean(queue<Node>& snake) {
  g_enter_direction = 'a';
  g_snake_direction = 'a';
  g_is_produce_food = 1;
  g_is_del_tail = 1;
  while (!snake.empty()) snake.pop();
}