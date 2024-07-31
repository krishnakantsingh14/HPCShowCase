#include "linkedlist.h"

LinkedList::LinkedList()
{
    head = nullptr;
}

void LinkedList::insert(int value) {
    Node* newNode = new Node(value) ;
    if (head==nullptr) {
        head = newNode;
    }
    else {
        Node* temp = head;
        while (temp->next != nullptr) {
            temp = temp->next;
        }
        temp->next = newNode;

    }

}

void LinkedList::deleteNode(int value) {
    if (head==nullptr) return;
    else {
        if (head->data == value ) {
            Node *temp = head;
            head = temp->next;
            delete temp;
            return;
            // temp = temp->next;
        }
    }
}