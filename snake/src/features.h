#ifndef FEATURES_H
#define FEATURES_H
#include <conio.h>
#include <windows.h>

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <queue>

#include "data.h"

//打印地图
void GameMap();

//移动光标
void MoveCursor(int xpoint, int ypoint);

//游戏结束
bool GameOver(int speed);

//速度设置
void SpeedSetting(int& speed);

//欢迎界面
void Welcome(int& speed);

//随机产生食物
void CreateFood();

//游戏初始化
void GameInit(std::queue<Node>& snake);

//移动蛇
void MoveSnake(std::queue<Node>& snake);

//蛇长大
void SnakeGrowUp(std::queue<Node>& snake);

//蛇死亡
bool IsSnakeDie(std::queue<Node>& snake);

//清除
void Clean(std::queue<Node>& snake);
#endif