#include "../include/data_json.h"

#include <fstream>
#include <iostream>
#include <string>

void LoadJson(List* list) {
  Json::Reader reader;
  Json::Value root;

  std::ifstream in("data//stu.json", std::ios::binary);

  if (!in.is_open()) {
    std::cout << "Open_Error!" << std::endl;
    return;
  }

  if (reader.parse(in, root)) {
    const Json::Value stu_arr = root["info"];
    Data data;
    for (unsigned int i = 0; i < stu_arr.size(); i++) {
      data.name = stu_arr[i]["name"].asString();
      data.age = stu_arr[i]["age"].asString();
      data.sex = stu_arr[i]["sex"].asString();
      InsertNodeTail(list, data);
    }
  } else {
    std::cout << "Parse_Error!" << std::endl;
  }
  in.close();
}

void WriteJson(List* list) {
  Json::Value root;
  root["info"];
  Json::Value inf;
  List* pmove = list->next;

  while (pmove) {
    inf["name"] = Json::Value(pmove->data.name);
    inf["age"] = Json::Value(pmove->data.age);
    inf["sex"] = Json::Value(pmove->data.sex);
    root["info"].append(inf);
    pmove = pmove->next;
  }

  Json::StyledWriter sw;
  std::ofstream os;
  os.open("data//stu.json.tmp", std::ios::out);
  if (!os.is_open()) std::cout << "Open_Error!" << std::endl;
  os << sw.write(root);
  // std::cout << "Saving......" << std::endl;
  rename("data//stu.json.tmp", "data//stu.json");
  // std::cout << "Saved" << std::endl;
  os.close();
}
