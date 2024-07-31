#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#pragma once
#include "Node.h"

class LinkedList
{



public:
    LinkedList();
    void insert(int value);
    void deleteNode(int value);

    void dispaly();
private:
    Node * head;
};

#endif