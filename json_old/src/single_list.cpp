#include "../include/single_list.h"

#include <string>

List* pmove;

//打印链表
void PrintList(List* list) {
  if (!list->next) return;
  pmove = list->next;
  while (pmove) {
    std::cout << pmove->data.name << " " << pmove->data.age << " "
              << pmove->data.sex << std::endl;
    pmove = pmove->next;
  }
}

//产生新节点
List* CreateNode(const Data data) {
  List* node = new List;
  node->data = data;
  node->next = NULL;
  return node;
}

List* CreateNode() {
  List* node = new List;
  node->next = NULL;
  return node;
}

//头插法
void InsertNodeHead(List* list, const Data data) {
  List* node = CreateNode(data);
  node->next = list->next;
  list->next = node;
}

//尾插法
void InsertNodeTail(List* list, const Data data) {
  while (list) {
    if (!list->next) {
      list->next = CreateNode(data);
      return;
    }
    list = list->next;
  }
}

//按值删除(遇到的第一个)
void DeleteNodeFirst(List* list, const Data data) {
  pmove = list->next;
  while (pmove->data.name != data.name) {
    list = pmove;
    pmove = pmove->next;
    if (!pmove) return;
  }
  list->next = pmove->next;
  delete pmove;
}

//按值删除(删除所有)
void DeleteNodeAll(List* list, const Data data) {
  pmove = list->next;
  while (pmove) {
    if (pmove->data.name == data.name) {
      list->next = pmove->next;
      delete pmove;
      pmove = list->next;
    } else {
      list = pmove;
      pmove = pmove->next;
    }
  }
}

//删除链表
void DeleteList(List* list) {
  while (list) {
    pmove = list;
    list = list->next;
    delete pmove;
    pmove = NULL;
  }
}
