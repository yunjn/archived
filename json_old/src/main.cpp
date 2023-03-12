#include "../include/data_json.h"
#include "../include/single_list.h"
using std::cin;
using std::cout;
using std::endl;
int main() {
  List* list = CreateNode();
  LoadJson(list);
  PrintList(list);
  cout << endl;
  Data stu;
  /*for (unsigned int i = 0; i < 10; i++) {
    cout << "name: ";
    cin >> stu.name;
    cout << "age: ";
    cin >> stu.age;
    cout << "sex: ";
    cin >> stu.sex;
    InsertNodeTail(list, stu);
    system("clear");
  }*/
  stu.name = "刘海柱";
  DeleteNodeFirst(list, stu);
  PrintList(list);
  stu.name = "刘海柱";
  stu.age = "19";
  stu.sex = "近战法师";
  InsertNodeHead(list, stu);
  WriteJson(list);
  DeleteList(list);
  return 0;
}
