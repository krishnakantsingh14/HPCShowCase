#ifndef NODE_H
#define NODE_H

#pragma once

class Node
{
public:
    int data ;
    Node * next;
    Node(int value);
    virtual ~Node() = default;

private:

};

#endif