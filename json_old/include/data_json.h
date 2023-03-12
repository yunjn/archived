/**************************
 * JSON
 * ************************/
#ifndef DATA_JSON_H
#define DATA_JSON_H
#include <jsoncpp/json/json.h>

#include "single_list.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

void LoadJson(List* list);

void WriteJson(List* list);

#endif