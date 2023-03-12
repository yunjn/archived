/**************************
 * 单链表的基本操作
 * ************************/
#ifndef SINGLE_LIST_H
#define SINGLE_LIST_H

#include <iostream>
typedef struct Dat {
  std::string name;
  std::string age;
  std::string sex;
} Data;

typedef struct Node {
  Data data;
  struct Node* next;
} List;

//打印链表
void PrintList(List* list);

//产生新节点
List* CreateNode(const Data data);
List* CreateNode();

//头插法
void InsertNodeHead(List* list, const Data data);

//尾插法
void InsertNodeTail(List* list, const Data data );

//按值删除(遇到的第一个)
void DeleteNodeFirst(List* list, const Data data);

//按值删除(删除所有)
void DeleteNodeAll(List* list, const Data data);

//删除链表
void DeleteList(List* list);

#endif