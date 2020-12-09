//
//  LRU.cpp
//  LeetCodeCPlusPlus
//
//  Created by Imp on 2020/11/19.
//  Copyright © 2020 imp. All rights reserved.
//

#include "LRU.hpp"
#include <unordered_map>

using namespace std;

class LRUCache {
    struct Node {
        int key;
        int val;
        Node *pre;
        Node *next;
        Node(){}
        Node(int k, int v) : key(k), val(v), pre(NULL), next(NULL) {}
    };
    int cap;
    Node *head;
    Node *tail;
    unordered_map<int, Node*> map;

    void moveToTail(Node *node) {
        node->pre = tail->pre;
        tail->pre = node;
        node->pre->next = node;
        node->next = tail;
    }

public:
    LRUCache(int capacity) {
        cap = capacity;
        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->pre = head;
    }

    int get(int key) {
        int res = -1;
        if (map.count(key)) {
            Node *cur = map[key];
            res = cur->val;
            cur->pre->next = cur->next;
            cur->next->pre = cur->pre;
            moveToTail(cur);
        }
        return res;
    }

    void put(int key, int value) {
        if (get(key) != -1) {
            map[key]->val = value;
            return;
        }
        Node *node = new Node(key,value);
        moveToTail(node);
        map[key] = node;
        if (map.size() >= cap) {
            map.erase(head->next->key);
            head->next = head->next->next;
            head->next->pre = head;
        }
    }
};
