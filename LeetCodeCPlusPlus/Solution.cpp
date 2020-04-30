//
//  Solution.cpp
//  LeetCodeCPlusPlus
//
//  Created by Imp on 2019/1/9.
//  Copyright © 2019 imp. All rights reserved.
//

#include "Solution.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctype.h>
#include <set>
#include <unordered_set>
#include <queue>
#include <stack>
#include <algorithm>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct Interval {
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
};

class Solution {
public:

    //2.两数相加
    /*
     如果l1和l2一个比较长 一个比较短就可能会浪费很多时间，可以同时判断l1,l2都不为空相加，然后再把不为空的那个节点加在后面
     */
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *res = NULL;
        ListNode *p = NULL;
        int up = 0;
        while (l1 || l2 || up) {
            int sum = up;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            if (sum >= 10) {
                up = 1;
            } else {
                up = 0;
            }
            ListNode *node = new ListNode(sum % 10);
            if (res == NULL) {
                res = node;
            } else {
                p->next = node;
            }
            p = node;
        }
        return res;
    }

    //3.无重复字符的最长子串
    int lengthOfLongestSubstring(string s) {
        if (s.length() <= 1) {
            return (int)s.length();
        }
        map<char,int> map;
        int left = 0;
        int res = 0;
        for (int i = 0; i < (int)s.length(); i++) {
            int index = -1;
            if (map.find(s[i]) != map.end()) {
                index = map[s[i]];
            }
            if (index >= left) {
                left = index + 1;
            } else if (res < (i - left + 1)) {
                res = i - left + 1;
            }
            map[s[i]] = i;
        }
        return  res;
    }

    //12.整数转罗马数字
    string intToRoman(int num) {
        map<int,string> map = {{1,"I"},{2,"II"},{3,"III"},{4,"IV"},{5,"V"},{6,"VI"},{7,"VII"},{8,"VIII"},{9,"IX"},{10,"X"},{20,"XX"},{30,"XXX"},{40,"XL"},{50,"L"},{60,"LX"},{70,"LXX"},{80,"LXXX"},{90,"XC"},{100,"C"},{200,"CC"},{300,"CCC"},{400,"CD"},{500,"D"},{600,"DC"},{700,"DCC"},{800,"DCCC"},{900,"CM"},{1000,"M"},{2000,"MM"},{3000,"MMM"}};
        string res = "";
        int div = 1000;
        while (num > 0 && div > 0) {
            int a = num / div;
            if (a > 0) {
                res += map[a * div];
                num %= div;
            }
            div /= 10;
        }
        return res;
    }

    //13.罗马数字转整数
    int romanToInt(string s) {
        int res = 0;
        map<char,int> maps = {{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
        for (int i = 0; i < s.length(); i++) {
            char a = s[i];
            res += maps[a];
            if (i > 0 && (((a == 'V' || a == 'X') && s[i-1] == 'I') || ((a == 'L' || a == 'C') && s[i-1] == 'X') || ((a == 'D' || a == 'M') && s[i-1] == 'C'))) {
                res -= 2 * maps[s[i-1]];
            }
        }
        return res;
    }

    //17.电话号码的字母组合
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        backTraceLetter(res, "", 0, digits);
        return res;
    }

    void backTraceLetter(vector<string>& res, string s, int index, string digits) {
        if (index >= digits.size()) {
            res.push_back(s);
            return;
        }
        vector<char> tmp = digitsToStrings(digits[index]);
        for (int j = 0; j < tmp.size(); j++) {
            s.push_back(tmp[j]);
            backTraceLetter(res, s, index + 1, digits);
            s.pop_back();
        }
    }

    vector<char> digitsToStrings(char s) {
        if (s == '2') {
            return vector<char>{'a','b','c'};
        } else if (s == '3') {
            return vector<char>{'d','e','f'};
        } else if (s == '4') {
           return vector<char>{'g','h','i'};
        } else if (s == '5') {
           return vector<char>{'j','k','l'};
        } else if (s == '6') {
            return vector<char>{'m','n','o'};
        } else if (s == '7') {
           return vector<char>{'p','q','r','s'};
        } else if (s == '8') {
           return vector<char>{'t','u','v'};
        } else if (s == '9') {
           return vector<char>{'w','x','y','z'};
        }
        return vector<char>{};
    }

    //19. 删除链表的倒数第N个节点
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *h = new ListNode(0);
        h->next = head;
        ListNode *low = h;
        ListNode *fast = h;
        while (n>-1 && fast != NULL) {
            fast = fast->next;
            n--;
        }
        while (fast) {
            low = low->next;
            fast = fast->next;
        }
        low->next = low->next->next;
        return h->next;
    }

    //20.有效的括号
    bool isValid(string s) {
        stack<char> left = stack<char>();
        for (int i = 0; i < s.length(); i++) {
            char sl = s[i];
            if (sl == '[' || sl == '{' || sl == '(') {
                left.push(sl);
            } else {
                if (left.empty()) {
                    left.push(sl);
                } else {
                    if ((sl == ']' && left.top() == '[') || (sl == '}' && left.top() == '{') || (sl == ')' && left.top() == '(')) {
                        left.pop();
                    } else {
                        left.push(sl);
                    }
                }
            }
        }
        return left.empty();
    }

    //22.括号生成
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        dfsGenerateParenthesis(res, 0, 0, "", n);
        return res;
    }

    void dfsGenerateParenthesis(vector<string> &res, int count1, int count2, string s, int n) {
        if (count2 > count1 || count1 > n || count2 > n) {
            return;
        }
        if (count2 == count1 && count1 == n) {
            res.push_back(s);
        }
        dfsGenerateParenthesis(res, count1+1, count2, s+'(', n);
        dfsGenerateParenthesis(res, count1, count2+1, s+')', n);
    }

    //23.合并k个排序链表
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *res = new ListNode(0);
        res->next = NULL;
        ListNode *cur = res;
        for (auto it = lists.begin(); it != lists.end(); ) {
            if (*it == NULL) {
                lists.erase(it);
            } else {
                it++;
            }
        }
        while (lists.size()) {
            int index = indexOfMInListNode(lists);
            ListNode *node = lists[index];
            cur->next = node;
            if (node->next) {
                lists[index] = node->next;
            } else {
                lists.erase(std::begin(lists)+index);
            }
            cur = cur->next;
        }
        cur->next = NULL;
        return res->next;
    }

    int indexOfMInListNode(vector<ListNode *> &lists) {
        int index = 0;
        int minNum = lists[0]->val;
        for (int i = 0; i < lists.size(); i++) {
            ListNode *node = lists[i];
            if (node->val < minNum) {
                index = i;
                minNum = node->val;
            }
        }
        return index;
    }

    //24. 两两交换链表中的节点
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }
        ListNode *tmp = new ListNode(0);
        tmp->next = head;
        head = tmp;
        while (head->next && head->next->next) {
            ListNode *p = head->next;
            ListNode *q = head->next->next;
            head->next = q;
            p->next = q->next;
            q->next = p;
            head = p;
        }
        return tmp->next;
    }

    //33.搜索排序旋转数组
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left)/2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < nums[right]) {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }

    //35.搜索插入位置
    int searchInsert(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] >= target) {
                return i;
            }
        }
        return (int)nums.size();
    }

    //42.接雨水
    int trap(vector<int>& height) {
        // left[i]表示i左边的最大值，right[i]表示i右边的最大值
        int n = (int)height.size();
        vector<int> left(n), right(n);
        for (int i = 1; i < n; i++) {
            left[i] = max(left[i - 1], height[i - 1]);
        }
        for (int i = n - 2; i >= 0; i--) {
            right[i] = max(right[i + 1], height[i + 1]);
        }
        int water = 0;
        for (int i = 0; i < n; i++) {
            int level = min(left[i], right[i]);
            water += max(0, level - height[i]);
        }
        return water;
    }

    //46.全排列
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        allPermute(res, nums, 0);
        return res;
    }

    void allPermute(vector<vector<int>> &res, vector<int> &v, int start) {
        if (start == v.size() - 1) {
            res.push_back(v);
        } else {
            for (int i = start; i < v.size(); i++) {
                int temp = v[start];
                v[start] = v[i];
                v[i] = temp;
                allPermute(res, v, start+1);
                //把第该层子序列第一个位置的值换成另外一个值，所以要交换回来
                temp = v[start];
                v[start] = v[i];
                v[i] = temp;
            }
        }
    }

    //48.旋转图像
    void rotate(vector<vector<int>>& matrix) {
        if (matrix.size()<2) {
            return;
        }
        long size = matrix.size();
        for (int i = 0; i < size / 2; i++) {
            for (int j = i; j < size - i - 1; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[size-1-j][i];
                matrix[size-1-j][i] = matrix[size-1-i][size-1-j];
                matrix[size-1-i][size-1-j] = matrix[j][size-1-i];
                matrix[j][size-1-i] = tmp;
            }
        }
    }

    //50.计算x的n次幂
    double myPow(double x, int n) {
        if (n == 0) return 1;
        double half = myPow(x, n / 2);
        if (n % 2 == 0) return half * half;
        else if (n > 0) return half * half * x;
        else return half * half / x;
    }

    //51.N皇后
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> queen(n,string(n,'.'));
        backtraceQueen(res, queen, 0);
        return res;
    }

    void backtraceQueen(vector<vector<string>>& res, vector<string>& queen, int row) {
        if (row == queen.size()) {
            res.push_back(queen);
            return;
        }
        for (int col = 0; col < queen.size(); col++) {
            if (isValidQueen(queen, col, row)) {
                queen[row][col] = 'Q';
                backtraceQueen(res, queen, row + 1);
                queen[row][col] = '.';
            }
        }
    }

    bool isValidQueen(vector<string>& queen, int col, int row) {
        //判断当前列是否有其他q
        for (int i = 0; i < row; i++) {
            if (queen[i][col] == 'Q') {
                return false;
            }
        }
        //判断当前位置的右上角区域是否有其他q
        for (int i = row - 1, j = col + 1; i >= 0 && j < queen.size(); i--,j++) {
            if (queen[i][j] == 'Q') {
                return false;
            }
        }
        //判断当前位置的左上角区域是否有其他q
        for (int i = row - 1,j = col - 1; i >= 0 && j >= 0; i--,j--) {
            if (queen[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
    //52.N皇后II
    //就是51返回的count
    int totalNQueens(int n) {
        return solveNQueens(n).size();
    }

    //53.最大子序和
    int maxSubArray(vector<int>& nums) {
        int res = nums[0];
        int sum = nums[0];
        for (int i = 1;i < nums.size();i++) {
            if (sum <= 0) {
                sum = nums[i];
            } else {
                sum += nums[i];
            }
            res = max(res,sum);
        }
        return res;
    }

    //54.螺旋矩阵1
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        int top = 0, bottom = (int)matrix.size() - 1;
        if (bottom < 0) return ans;
        int left = 0, right = (int)matrix[0].size() - 1;
        int i = 0, j = 0;
        while (true) {
            for (j = left; j <= right; j++) ans.push_back(matrix[i][j]);
            top++;
            j--;
            if (top > bottom) break;
            for (i = top; i <= bottom; i++) ans.push_back(matrix[i][j]);
            right--;
            i--;
            if (right < left) break;
            for (j = right; j >= left; j--) ans.push_back(matrix[i][j]);
            bottom--;
            j++;
            if (top > bottom) break;
            for (i = bottom; i >= top; i--) ans.push_back(matrix[i][j]);
            left++;
            i++;
            if (right < left) break;
        }
        return ans;
    }

    //55.跳跃游戏
    bool canJump(vector<int>& nums) {
        if (nums.size() < 2) {
            return true;
        }
        int maxIndex = nums[0];
        for (int i = 1; i < nums.size() - 1; i++) {
            if (i <= maxIndex) {
                maxIndex = max(maxIndex, nums[i] + i);
            }
        }
        return maxIndex >= nums.size() - 1;
    }

    //56.合并区间
    vector<Interval> merge(vector<Interval>& intervals) {
        if (intervals.size() <= 1) {
            return intervals;
        }
        vector<Interval> res;
        sort(intervals.begin(), intervals.end(), [](Interval x, Interval y){return x.start < y.start;});
        int i = 0;
        Interval tmp = intervals[0];
        while(i < intervals.size()){
            if(i + 1 < intervals.size() && tmp.end >= intervals[i+1].start){
                if(tmp.end < intervals[i+1].end){
                    tmp.end = intervals[i+1].end;
                }
            }else{
                res.push_back(tmp);
                tmp = intervals[i+1];
            }
            ++i;
        }
        return res;
    }

    //59.螺旋矩阵2
    vector<vector<int>> generateMatrix(int n) {
        int s = 0;
        int e = n - 1;
        int num = 1;
        vector<vector<int>> res(n, vector<int>(n, 0)); // 必须初始化
        while(s < e) {
            for(int j = s; j <= e; j++) res[s][j] = num++;
            for(int i = s + 1; i <= e; i++) res[i][e] = num++;
            for(int j = e - 1; j >= s; j--) res[e][j] = num++;
            for(int i = e - 1; i > s; i--) res[i][s] = num++;
            ++s;
            --e;
        }
        if(s == e) res[s][s] = num;
        return res;
    }

    //61. 旋转链表
    ListNode* rotateRight(ListNode* head, int k) {
        if (head == NULL || k == 0) {
            return head;
        }
        ListNode *tmp = head;
        ListNode *q = head;
        int count = 1;
        while (tmp->next) {
            tmp = tmp->next;
            count++;
        }
        if (k % count == 0) {
            return head;
        }
        int len = count - k % count;
        while (--len) {
            q = q->next;
        }
        tmp->next = head;
        head = q->next;
        q->next = NULL;
        return head;
    }

    //67.二进制求和
    string addBinary(string a, string b) {
        string res;
        int flag = 0;
        long lenghtA = a.length();
        long lenghtB = b.length();
        long lenght = max(lenghtA, lenghtB);
        for (int i = 0; i < lenght; i++) {
            int numA = 0;
            int numB = 0;
            if (lenghtA - i - 1 >= 0) {
                numA = binaryCharStringToInt(a[lenghtA - i - 1]);
            }
            if (lenghtB - i - 1 >= 0) {
                numB = binaryCharStringToInt(b[lenghtB - i - 1]);
            }
            int sum = numA + numB + flag;
            string r = "0";
            flag = 0;
            if (sum == 1) {
                r = "1";
            } else if (sum == 2) {
                flag = 1;
            } else if (sum == 3) {
                flag = 1;
                r = "1";
            }
            res.insert(0, r);
        }
        if (flag == 1) {
            res.insert(0, "1");
        }
        return res;
    }

    int binaryCharStringToInt(char c) {
        if (c == '1') {
            return 1;
        } else {
            return 0;
        }
    }

    //68.文本左右对齐
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        int index = 0;
        int rowSize = 0;
        vector<string> temp;
        while (index < words.size()) {
            string single = words[index];
            if (rowSize == 0) {
                temp.push_back(single);
                rowSize += single.size();
                index++;
            } else if (rowSize + single.size() + 1 <= maxWidth){
                temp.push_back(single);
                rowSize += single.size() + 1;
                index++;
            } else {
                string row;
                if (temp.size() == 1) {
                    row = temp[0];
                    row += blankCount(maxWidth - temp[0].size());
                } else {
                    unsigned long blank = maxWidth - (rowSize - (temp.size() - 1));
                    unsigned long count = blank / (temp.size() - 1);
                    unsigned long div = blank % (temp.size() - 1);
                    for (int i = 0; i < temp.size(); i++) {
                        row += temp[i];
                        unsigned long newCount = count;
                        if (div > 0) {
                            newCount++;
                            div--;
                        }
                        if (i != temp.size() - 1) {
                            row += blankCount(newCount);
                        }
                    }
                }
                res.push_back(row);
                temp.clear();
                rowSize = 0;
            }
        }
        if (temp.size()) {
            string row;
            for (int i = 0; i < temp.size(); i++) {
                row += temp[i];
                if (i != temp.size() - 1) {
                    row += " ";
                }
            }
            if (rowSize < maxWidth) {
                row += blankCount(maxWidth - rowSize);
            }
            res.push_back(row);
        }
        return res;
    }

    string blankCount(unsigned long k) {
        string res;
        while (k > 0) {
            res += " ";
            k--;
        }
        return res;
    }

    //71.简化路径
    string simplifyPath(string path) {
        path += "/";
        vector<string> s;
        string res = "/";
        int index = 1;
        string dir;
        while (index < path.length()) {
            if (path[index] != '/') {
                dir += path[index];
            } else {
                if (dir.length()) {
                    if (dir == ".." && !s.empty()) {
                        s.pop_back();
                    } else if (dir != "." && dir != "..") {
                        s.push_back(dir);
                    }
                }
                dir = "";
            }
            index++;
        }
        for (auto it = s.begin(); it != s.end(); it++) {
            res += *it;
            res += "/";
        }
        if (res.length() != 1) {
            res.erase(res.end() - 1);
        }
        return res;
    }

    //74.搜索二维矩阵
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.size() == 0) {
            return false;
        }
        if (matrix[0].size() == 0) {
            return false;
        }
        int row = 0;
        for (int i = 0; i < matrix.size(); i++) {
            if (matrix[i][0] > target) {
                break;
            }
            if (matrix[i][0] == target) {
                return true;
            }
            row = i;
        }
        unsigned long col = matrix[0].size();
        if (matrix[row][col-1] < target) {
            return false;
        }
        unsigned long left = 0,right = col - 1;
        while (left <= right) {
            int index = left + (right - left)/2;
            if (matrix[row][index] == target) {
                return true;
            } else if (matrix[row][index] > target) {
                right = index-1;
            } else {
                left = index+1;
            }
        }
        return false;
    }

    //78.找子集
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> tmp;
        vector<vector<int>> res;
        backtrace(res, nums, tmp, 0);
        return res;
    }

    void backtrace(vector<vector<int>> &res, vector<int> &nums, vector<int> &tmp, int index) {
        res.push_back(tmp);
        for (int i = index; i < nums.size(); i++) {
            tmp.push_back(nums[i]);
            backtrace(res, nums, tmp, i+1);
            tmp.pop_back();
        }
    }

    //80.删除排序数组中重复项||
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        int target = nums[0];
        int count = 1;
        for (auto it = nums.begin() + 1; it != nums.end(); ) {
            if (*it == target) {
                count++;
                if (count > 2) {
                    nums.erase(it);
                } else {
                    it++;
                }
            } else {
                target = *it;
                count = 1;
                it++;
            }
        }
        return nums.size();
    }

    //82. 删除排序链表中的重复元素||
    /*
     有多个连续的话 需要判断完 并不一定只有22连续的
     */
    ListNode* deleteDuplicates2(ListNode* head) {
        ListNode* p = new ListNode(0);
        p->next = head;
        head = p;
        ListNode *left,*right;
        while(p->next)
        {
            left = p->next;
            right = left;
            while(right->next && right->next->val == left->val) {
                right = right->next;
            }
            if(left == right) {
                p = p->next;
            } else {
                p->next = right->next;
            }
        }
        return head->next;
    }

    //83. 删除排序链表中的重复元素
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode *p = head;
        while (p && p->next) {
            if (p->val == p->next->val) {
                p->next = p->next->next;
            }else {
                p = p->next;
            }
        }
        return head;
    }

    //86.分割链表
    ListNode* partition(ListNode* head, int x) {
        ListNode *less = new ListNode(0);
        ListNode *greater = new ListNode(0);
        ListNode *curLess = less;
        ListNode *curGreater = greater;
        while (head) {
            if (head->val < x) {
                curLess->next = head;
                curLess = curLess->next;
            } else {
                curGreater->next = head;
                curGreater = curGreater->next;
            }
            head = head->next;
        }
        curGreater->next = nullptr;
        curLess->next = greater->next;
        return less->next;
    }

    //90.子集II
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tmp;
        sort(nums.begin(), nums.end());
        subsetsWithDupBacktrace(res, tmp, nums, 0);
        return res;
    }

    void subsetsWithDupBacktrace(vector<vector<int>> &res, vector<int> &tmp, vector<int> &nums, int index) {
        res.push_back(tmp);
        for (int i = index; i < nums.size(); i++) {
            if (i!=index && nums[i-1]==nums[i]) {
                continue;
            }
            tmp.push_back(nums[i]);
            subsetsWithDupBacktrace(res, tmp, nums, i+1);
            tmp.pop_back();
        }
    }

    //91.解码方法
    int numDecodings(string s) {
        if (s.length() == 0 || (s.length() == 1 && s[0] == '0')) {
            return 0;
        }
        if (s.length() == 1) {
            return 1;
        }
        vector<int> dp(s.length() + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '0') {
                dp[i+1] = 0;
            } else {
                dp[i+1] = dp[i];
            }
            if (i > 0 && (s[i-1] == '1' || (s[i-1] == '2' && s[i] <= '6'))) {
                dp[i+1] += dp[i-1];
            }
        }
        return dp.back();
    }

    //92. 反转链表 II
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if (m == n) {
            return head;
        }
        ListNode *temp = new ListNode(0);
        temp->next = head;
        ListNode *pre = temp;
        ListNode *tail = nullptr;
        for (int i = 1; i <= n; i++) {
            if (i < m) {
                pre = pre->next;
            } else if (i == m) {
                tail = pre->next;
            } else {
                ListNode *node = tail->next;
                tail->next = tail->next->next;
                node->next = pre->next;
                pre->next = node;
            }
        }
        return temp->next;
    }

    //94.二叉树的中序遍历
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> v = vector<int>();
        inorderRecursionTraversal(v, root);
        return v;
    }
    //递归
    void inorderRecursionTraversal(vector<int> &v, TreeNode *node) {
        if (node == NULL) {
            return;
        }
        inorderRecursionTraversal(v, node->left);
        v.push_back(node->val);
        inorderRecursionTraversal(v, node->right);
    }
    //非递归
    void inorderNoRecursionTraversal(vector<int> &v, TreeNode *node) {
        stack<TreeNode *> s;
        while (!s.empty() || node != NULL) {
            if (node) {
                s.push(node);
                node = node->left;
            } else {
                node = s.top();
                s.pop();
                v.push_back(node->val);
                if (node->right != NULL) {
                    s.push(node->right);
                }
            }
        }
    }

    //95.不同的二叉搜索树II
    vector<TreeNode*> generateTrees(int n) {
        if (n == 0) {
            return vector<TreeNode *>{};
        }
        vector<TreeNode *>res = numSectionToTree(1, n);
        return res;
    }

    vector<TreeNode *>numSectionToTree(int left, int right) {
        vector<TreeNode *>res;
        if (left > right) {
            res.push_back(nullptr);
            return res;
        }
        for (int i = left; i <= right; i++) {
            vector<TreeNode *>leftNodes = numSectionToTree(left, i-1);
            vector<TreeNode *>rightNodes = numSectionToTree(i+1, right);
            for(TreeNode *leftNode : leftNodes) {
                for (TreeNode *rightNode : rightNodes) {
                    TreeNode *node = new TreeNode(i);
                    node->left = leftNode;
                    node->right = rightNode;
                    res.push_back(node);
                }
            }
        }
        return res;
    }

    //96.不同的二叉搜索树
    int numTrees(int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return 1;
        }
        vector<int> dp = vector<int>(n+1);
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j]*dp[i-j-1];
            }
        }
        return dp[n];
    }

    //97.交错字符串
    bool isInterleave(string s1, string s2, string s3) {
        int len1 = (int)s1.length();
        int len2 = (int)s2.length();
        if (s3.length() != s1.length() + s2.length()) {
            return false;
        }
        vector<vector<bool>> dp(len1 + 1, vector<bool>(len2 + 1, true));
        for(int i=1; i<=len1; i++) {
            dp[i][0] = (s1[i-1]==s3[i-1])&&dp[i-1][0];
        }
        for(int j=1;j<=len2; j++) {
            dp[0][j] = (s2[j-1]==s3[j-1])&&dp[0][j-1];
        }
        for(int i=1; i<=len1; i++) {
            for(int j=1; j<=len2; j++) {
                dp[i][j] = ((s1[i-1]==s3[i+j-1])&&dp[i-1][j]) || ((s2[j-1]==s3[i+j-1])&&dp[i][j-1]);
            }
        }
        return dp[len1][len2];
    }

    //99.恢复二叉搜索树
    void recoverTree(TreeNode* root) {
        vector<TreeNode *> res = vector<TreeNode *>{};
        midTraverse(&res, root);
        TreeNode *node1 = nullptr;
        TreeNode *node2 = nullptr;
        for (int i = 0; i < res.size()-1; i++) {
            if (res[i]->val > res[i+1]->val && node1 == nullptr) {
                node1 = res[i];
                node2 = res[i+1];
            } else if (res[i]->val > res[i+1]->val && node1 != nullptr) {
                node2 = res[i+1];
            }
        }
        int val = node1->val;
        node1->val = node2->val;
        node2->val = val;
    }

    void midTraverse(vector<TreeNode *> *res, TreeNode *root) {
        if (root == nullptr) {
            return;
        }
        midTraverse(res, root->left);
        res->push_back(root);
        midTraverse(res, root->right);
    }

    //102.二叉树的层次遍历
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> v;
        if (root == NULL) {
            return v;
        }
        queue<TreeNode *> parentS;
        parentS.push(root);
        queue<TreeNode *> childS;
        while (!parentS.empty() || !childS.empty()) {
            vector<int> vc;
            while (!parentS.empty()) {
                root = parentS.front();
                if (root -> left != NULL) childS.push(root->left);
                if (root -> right != NULL) childS.push(root->right);
                vc.push_back(root->val);
                parentS.pop();
            }
            parentS = childS;
            childS = queue<TreeNode *>();
            v.push_back(vc);
        }
        return v;
    }

    //103. 二叉树的锯齿形层次遍历
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>>v;
        if (root == nullptr) {
            return v;
        }
        bool isLeft = false;
        queue<TreeNode *>parents;
        parents.push(root);
        queue<TreeNode *>childs;
        while (!parents.empty() || !childs.empty()) {
            vector<int> res;
            while (!parents.empty()) {
                root = parents.front();
                if (root->left) {
                    childs.push(root->left);
                }
                if (root->right) {
                    childs.push(root->right);
                }
                res.push_back(root->val);
                parents.pop();
            }
            if (isLeft) {
                ::reverse(res.begin(),res.end());
            }
            isLeft = !isLeft;
            parents = childs;
            childs = queue<TreeNode *>();
            v.push_back(res);
        }
        return v;
    }

    //105. 从前序与中序遍历序列构造二叉树
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return buildTreeFun(preorder, inorder, 0, preorder.size()-1, 0, inorder.size()-1);
    }

    TreeNode* buildTreeFun(vector<int>& preorder, vector<int>& inorder, int leftPre, int rightPre, int leftIn, int rightIn) {
        if (leftIn > rightIn || leftPre > rightPre) {
            return NULL;
        }
        TreeNode *root = new TreeNode(preorder[leftPre]);
        int rootIn = leftIn;
        while (rootIn <= rightIn && preorder[leftPre] != inorder[rootIn]) {
            rootIn++;
        }
        int left = rootIn - leftIn;
        root->left = buildTreeFun(preorder, inorder, leftPre + 1, leftPre + left, leftIn, rootIn - 1);
        root->right = buildTreeFun(preorder, inorder, leftPre + left + 1, rightPre, rootIn + 1, rightIn);
        return root;
    }

    //106.从后序与中序遍历序列构造二叉树
    TreeNode* buildTree2(vector<int>& inorder, vector<int>& postorder) {
        return bulidTreeFun2(inorder, postorder, 0, postorder.size()-1, 0, inorder.size()-1);
    }

    TreeNode* bulidTreeFun2(vector<int>& inorder, vector<int>& postorder, int leftP, int rightP, int leftIn, int rightIn) {
        if (leftIn > rightIn) {
            return NULL;
        }
        TreeNode *node = new TreeNode(postorder[rightP]);
        int rootIn = leftIn;
        while (rootIn <= rightIn && inorder[rootIn] != postorder[rightP]) {
            rootIn++;
        }
        int left = rootIn - leftIn;
        node->left = bulidTreeFun2(inorder, postorder, leftP, leftP+left-1, leftIn, rootIn-1);
        node->right = bulidTreeFun2(inorder, postorder, leftP+left, rightP-1, rootIn+1, rightIn);
        return node;
    }

    //107.二叉树层次遍历II 把102倒序
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> v;
        if (root == NULL) {
            return v;
        }
        queue<TreeNode *> parentS;
        parentS.push(root);
        queue<TreeNode *> childS;
        while (!parentS.empty() || !childS.empty()) {
            vector<int> vc;
            while (!parentS.empty()) {
                root = parentS.front();
                if (root -> left != NULL) childS.push(root->left);
                if (root -> right != NULL) childS.push(root->right);
                vc.push_back(root->val);
                parentS.pop();
            }
            parentS = childS;
            childS = queue<TreeNode *>();
            v.push_back(vc);
        }
        ::reverse(v.begin(),v.end());
        return v;
    }

    //108.将有序数组转换为二叉搜索树
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        int size = nums.size();
        if (size == 0) {
            return nullptr;
        }
        TreeNode *res = arrayToTree(nums, 0, size-1);
        return res;
    }

    TreeNode* arrayToTree(vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        int index = (left + right)/ 2;
        TreeNode *node = new TreeNode(nums[(left + right)/ 2]);
        node->left = arrayToTree(nums, left, index - 1);
        node->right = arrayToTree(nums, index+1, right);
        return node;
    }

    //109.将有序链表转换为二叉搜索树
    TreeNode* sortedListToBST(ListNode* head) {
        if (head == nullptr) {
            return nullptr;
        }
        if (head->next == nullptr) {
            return new TreeNode(head->val);
        }
        ListNode *newH = head;
        ListNode *slow = head->next;
        ListNode *fast = head->next->next;
        while (fast != nullptr && fast->next != nullptr) {
            newH = newH->next;
            slow = slow->next;
            fast = fast->next->next;
        }
        newH->next = nullptr;
        TreeNode *node = new TreeNode(slow->val);
        node->left = sortedListToBST(head);
        node->right = sortedListToBST(slow->next);
        return node;
    }

    //116.填充每个节点的下一个右侧节点指针
    Node* connect(Node* root) {
        nextNode(root);
        return root;
    }

    void nextNode(Node *root) {
        if (root == nullptr || root->left == nullptr) {
            return;
        }
        root->left->next = root->right;
        if (root->next) {
            root->right->next = root->next->left;
        }
        nextNode(root->left);
        nextNode(root->right);
    }

    //117.填充每个节点的下一个右侧节点指针II
    Node* connect2(Node* root) {
        nextNode(root);
        return root;
    }
    void nextNode2(Node *root) {
        if (root == nullptr) {
            return;
        }
        if (root->left != nullptr) {
            if (root->right) {
                root->left->next = root->right;
            } else if (root->next) {
                Node *trueNext = root->next;
                while (trueNext != nullptr) {
                    if (trueNext->left != nullptr) {
                        trueNext = trueNext->left;
                        break;
                    } else if (trueNext->right != nullptr) {
                        trueNext = trueNext->right;
                        break;
                    } else {
                        trueNext = trueNext->next;
                    }
                }
                root->left->next = trueNext;
            }
        }
        if (root->right != nullptr) {
            Node *trueNext = root->next;
            while (trueNext != nullptr) {
                if (trueNext->left != nullptr) {
                    trueNext = trueNext->left;
                    break;
                } else if (trueNext->right != nullptr) {
                    trueNext = trueNext->right;
                    break;
                } else {
                    trueNext = trueNext->next;
                }
            }
            root->right->next = trueNext;
        }
        nextNode2(root->right);
        nextNode2(root->left);
    }

    //120.三角形最短路径和
    int minimumTotal(vector<vector<int>>& triangle) {
        if (triangle.size() == 0) {
            return 0;
        }
        vector<vector<int>> dp;
        dp.push_back(vector<int>(1,triangle[0][0]));
        for (int i = 1; i < triangle.size(); i++) {
            vector<int> smallDp(triangle.size());
            for (int j = 0; j < triangle[i].size(); j++) {
                if (j == 0) {
                    smallDp[j] = dp[i-1][j] + triangle[i][j];
                } else if (j == triangle[i].size() - 1) {
                    smallDp[j] = dp[i-1][j-1] + triangle[i][j];
                } else {
                    smallDp[j] = min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j];
                }
            }
            dp.push_back(smallDp);
        }
        long size = dp.size();
        int res = dp[size-1][0];
        for (int i = 1; i < dp[size-1].size(); i++) {
            res = min(res, dp[size-1][i]);
        }
        return res;
    }

    //128.最长连续序列
    int longestConsecutive(vector<int>& nums) {
        map<int,int> map;
        int result = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (map.find(nums[i]) == map.end()) {
                int left = nums[i] - 1;
                int right = nums[i] + 1;
                int len1 = 0;
                if (map.find(left) != map.end()) {
                    len1 = map[left];
                }
                int len2 = 0;
                if (map.find(right) != map.end()) {
                    len2 = map[right];
                }
                int len = len1 + len2 + 1;
                result = max(len, result);
                map[nums[i]] = len;
                if (map.find(left) != map.end()) {
                    map[nums[i]-len1] = len;
                }
                if (map.find(right) != map.end()) {
                    map[nums[i]+len2] = len;
                }
            }
        }
        return result;
    }

    //129.求根到叶子节点数字之和
    int sumNumbers(TreeNode* root) {
        return sumNode(root, 0);
    }

    int sumNode(TreeNode *node, int sum) {
        if (node == nullptr) {
            return 0;
        } else if (!node->left && !node->right) {
            return sum * 10 + node->val;
        } else {
            return sumNode(node->left, sum * 10 + node->val) + sumNode(node->right, sum * 10 + node->val);
        }
    }

    //137.只出现一次的数字||
    int singleNumber(vector<int>& nums) {
        int a = 0, b = 0;
        for (auto x : nums) {
            a = (a ^ x) & ~b;
            b = (b ^ x) & ~a;
        }
        return a;
    }

    //141.环形链表
    /*
     * 快慢指针方法，如果有环，快指针最终会追上慢指针
     */
    bool hasCycle(ListNode *head) {
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast || slow) {
            if (!fast || !fast->next) {
                return false;
            }else if (fast == slow) {
                return true;
            }
            fast = fast->next->next;
            slow = slow->next;
        }
        return false;
    }

    //142.环形链表||
    /*
     先通过快慢指针找出是否有环  在相遇的点的同时从head出发一个指针再相遇的时候就是入口
     */
    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                slow = head;
                while (slow != fast) {
                    slow = slow->next;
                    fast = fast->next;
                }
                return slow;
            }
        }
        return NULL;
    }

    //143.重排链表
    void reorderList(ListNode* head) {
        if (head == NULL || head->next == NULL) {
            return;
        }
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode *newh = reserList(slow->next);
        slow->next = NULL;
        ListNode *temp = head;
        while (newh && temp) {
            ListNode *curSecond = newh;
            newh= newh->next;
            ListNode *nextCur = temp->next;
            curSecond->next = temp->next;
            temp->next = curSecond;
            temp = nextCur;
        }
    }

    ListNode *reserList(ListNode *head) {
        ListNode* newh = NULL;
        for(ListNode* p = head; p;) {
            ListNode* tmp = p -> next;
            p -> next = newh;
            newh = p;
            p = tmp;
        }
        return newh;
    }

    //144.二叉树前序遍历
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res = vector<int>();
        frontRecursionTraversal(res, root);
        return res;
    }

    //递归实现
    void frontRecursionTraversal(vector<int> &v ,TreeNode *node) {
        if (node == NULL) {
            return;
        }
        v.push_back(node->val);
        frontRecursionTraversal(v, node->left);
        frontRecursionTraversal(v, node->right);
    }
    //非递归实现
    void frontNoRecursionTraversal(vector<int> &v ,TreeNode *node) {
        if(node == NULL) return;
        stack<TreeNode*> s;
        //将头结点进栈
        s.push(node);
        while(!s.empty()){
            node = s.top();
            s.pop();
            v.push_back(node->val);
            //如果取出的节点的左右子树不为空，就将其压栈
            if(node->right != NULL) s.push(node->right);
            if(node->left != NULL) s.push(node->left);
        }
    }

    //145.二叉树的后序遍历
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> v;
        postorderRecursionTraversal(v, root);
        return v;
    }
    //递归
    void postorderRecursionTraversal(vector<int> &v, TreeNode *node) {
        if (node == NULL) return;
        postorderRecursionTraversal(v, node->left);
        postorderRecursionTraversal(v, node->right);
        v.push_back(node->val);
    }
    //非递归 前序把左右子树换下位置倒序
    void postorderNoRecursionTraversal(vector<int> &v, TreeNode *node) {
        stack<TreeNode *> s;
        s.push(node);
        while(!s.empty()){
            node = s.top();
            s.pop();
            if(node->left != NULL) s.push(node->left);
            if(node->right != NULL) s.push(node->right);
            v.push_back(node->val);
        }
        ::reverse(v.begin(),v.end());
    }

    //151.翻转字符串里的单词
    string reverseWords(string s) {
        vector<string> strs;
        string tmp;
        bool done = true;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                if (done) {
                    continue;
                } else {
                    done = true;
                    strs.push_back(tmp);
                    tmp = "";
                }
            } else {
                done = false;
                tmp += s[i];
            }
            if (i == s.size()-1 && tmp.size() > 0) {
                strs.push_back(tmp);
            }
        }
        string res;
        for (int i = strs.size()-1; i >= 0; i--) {
            res += strs[i];
            res += ' ';
        }
        if (res.size() > 0) {
            res.pop_back();
        }
        return res;
    }

    //160.相交链表
    //a+c+b = b+c+a长度
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA;
        ListNode *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }

    //174.地下城游戏
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int L = (int)dungeon[0].size(), row = (int)dungeon.size();
        vector<int> dp(L, 0);
        dp[L - 1] = max(1 - dungeon.back()[L - 1], 1);
        for (int i = L - 2; i >= 0; --i)
            dp[i] = max(dp[i + 1] - dungeon.back()[i], 1);
        for (int k = row - 2; k >= 0; --k) {
            dp[L - 1] = max(dp[L - 1] - dungeon[k][L - 1], 1);
            for (int i = L - 2; i >= 0; --i)
                dp[i] = max(min(dp[i + 1], dp[i]) - dungeon[k][i], 1);
        }
        return dp[0];
    }
    //190.颠倒二进制位
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res = res|(n&1);
            n >>= 1;
        }
        return res;
    }
    //191.位1的个数
    int hammingWeight(uint32_t n) {
        int res = 0;
        while (n > 0) {
            if (n & 1) {
                res += 1;
            }
            n>>=1;
        }
        return res;
    }

    //198.打家劫舍
    int rob(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        if (nums.size() == 1) {
            return nums[0];
        }
        vector<int> dp = vector<int>(nums.size());
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < nums.size(); i++) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }
        return dp[nums.size()-1];
    }

    //199.二叉树的右视图
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res = vector<int>();
        if (root == nullptr) {
            return res;
        }
        queue<TreeNode *> parents;
        parents.push(root);
        queue<TreeNode *> childs;
        while (!parents.empty() || !childs.empty()) {
            while (!parents.empty()) {
                TreeNode *top = parents.front();
                if (top->left) {
                    childs.push(top->left);
                }
                if (top->right) {
                    childs.push(top->right);
                }
                if (parents.size() == 1) {
                    res.push_back(parents.front()->val);
                }
                parents.pop();
            }
            parents = childs;
            childs = queue<TreeNode *>();
        }
        return res;
    }

    //201.数字范围按位与
    int rangeBitwiseAnd(int m, int n) {
        int bits = 0;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            ++bits;
        }
        return m << bits;
    }

    //202.快乐数
    bool isHappy(int n) {
        set<int> set;
        while (true) {
            n = getNextHappy(n);
            if (n == 1) {
                return true;
            } else if (set.find(n) != set.end()) {
                return false;
            } else {
                set.insert(n);
            }
        }
    }

    int getNextHappy(int n) {
        int res = 0;
        while (n > 0) {
            int tmp = n % 10;
            res += tmp * tmp;
            n /= 10;
        }
        return res;
    }

    //203.移除链表
    ListNode* removeElements(ListNode* head, int val) {
        ListNode *res = new ListNode(0);
        res->next = head;
        ListNode *cur = res;
        while (cur->next) {
            if (cur->next->val == val) {
                cur->next = cur->next->next;
            } else {
                cur = cur->next;
            }
        }
        return res->next;
    }
    //206.反转链表
    ListNode* reverseList(ListNode* head) {
        ListNode* newh = NULL;
        for(ListNode* p = head; p;) {
            ListNode* tmp = p -> next;
            p -> next = newh;
            newh = p;
            p = tmp;
        }
        return newh;
    }

    //213.打家劫舍
    int rob2(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        if (nums.size() == 1) {
            return nums[0];
        }
        vector<int> dp1 = vector<int>(nums.size());
        vector<int> dp2 = vector<int>(nums.size());
        dp1[0] = 0;
        dp1[1] = nums[0];
        dp2[0] = 0;
        dp2[1] = nums[1];
        for (int i = 2; i < nums.size(); i++) {
            dp1[i] = max(dp1[i-1], dp1[i-2] + nums[i-1]);
        }
        for (int i = 2; i < nums.size(); i++) {
            dp2[i] = max(dp2[i-1], dp2[i-2] + nums[i]);
        }
        return max(dp1[nums.size()-1], dp2[nums.size() - 1]);
    }

    //214.最短回文串
    bool Check(string s,int low,int high) {
        if(low==high)
            return true;
        while(low<high)
        {
            if(s[low]!=s[high])
                return false;
            low++;
            high--;
        }
        return true;
    }

    string shortest(string s) {
        int i;
        int len;
        string result="";
        len=(int)s.length()-1;

        if(len<=0)
            return "";

        for(;len>0;len--)  //从最后一个开 始往前找
        {
            if(s[0]==s[len]&&Check(s,0,len))
                break;
        }

        //找到后比如 0--len表示最长的回文，len--length()-1就是没有匹配上的，反转加在最前面就是
        for(i=(int)s.length()-1;i>len;i--)
            result+=s[i];

        result+=s;
        return result;
    }

    //224.基本计算器
    int calculate(string s) {
        int res=0;
        int sign=1;
        int n=int(s.size());
        stack<int> st;
        for(int i=0;i<n;++i){
            char c=s[i];
            if(c > '0'){
                int num=0;
                while(i<n && s[i]>='0'){
                    num=num*10+(s[i]-'0');
                    i++;
                }
                res=res+sign*num;
                i--;
            } else if(c=='+')
                sign=1;
            else if(c=='-')
                sign=-1;
            else if(c=='('){
                st.push(res);
                st.push(sign);
                res=0;
                sign=1;
            }
            else if(c==')'){
                res *= st.top(); st.pop();
                res += st.top(); st.pop();
            }
        }
        return res;
    }

    //226.翻转二叉树
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return nullptr;
        }
        TreeNode *left = invertTree(root->left);
        TreeNode *right = invertTree(root->right);
        root->left = right;
        root->right = left;
        return root;
    }

    //234.回文链表
    bool isPalindrome(ListNode* head) {
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
        }
        slow = reserList(slow);
        while (head && slow) {
            if (head->val != slow->val) {
                return false;
            }
            head = head->next;
            slow = slow->next;
        }
        return true;
    }

    //237.删除链表中节点
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }

    //289.生命游戏
    void gameOfLife(vector<vector<int>>& board) {
        vector<vector<int>> status(board.size(), vector<int>(board[0].size(), 0));
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (isChangeStatus(board, i, j)) {
                    status[i][j] = 1;
                }
            }
        }
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (status[i][j] == 1) {
                    board[i][j] = !board[i][j];
                }
            }
        }
    }

    bool isChangeStatus(vector<vector<int>> &borad, int i, int j) {
        bool res = false;
        int count = 0;
        for (int row = i - 1; row <= i + 1; row++) {
            for (int col = j - 1; col <= j+1; col++) {
                if (row == i && col == j) {
                    continue;
                }
                if (row < 0 || col < 0) {
                    continue;
                }
                if (row >= borad.size() || col >= borad[0].size()) {
                    continue;
                }
                if (borad[row][col] == 1) {
                    count++;
                }
            }
        }
        if (borad[i][j] == 1) {
            if (count < 2 || count > 3) {
                res = true;
            }
        } else {
            if (count == 3) {
                res = true;
            }
        }
        return res;
    }

    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        dp[0][0]=grid[0][0];
        for (int i = 1; i < n;i++) {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }
        for (int i = 1; i < m;i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for (int i = 1; i < m;i++) {
            for (int j = 1;j < n;j++) {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }

    //319.灯泡开关
    int bulbSwitch(int n) {
        int res = 0;
        for (int i = 1; i*i <= n; i++) {
            res++;
        }
        return res;
//        return sqrt(n);//这也可以
    }

    //328.奇偶链表
    ListNode* oddEvenList(ListNode* head) {
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *odd = new ListNode(0);
        ListNode *even = new ListNode(0);
        ListNode *curOdd = odd;
        ListNode *curEven = even;
        int index = 1;
        while (head) {
            if (index % 2 == 0) {
                curEven->next = head;
                curEven = curEven->next;
            } else {
                curOdd->next = head;
                curOdd = curOdd->next;
            }
            head = head->next;
            index += 1;
        }
        curOdd->next = even->next;
        curEven->next = NULL;//提交的时候不加这个会超时。。。
        return odd->next;
    }

    ListNode* oddEvenList2(ListNode* head) {
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *odd = head;
        ListNode *even = head->next;
        ListNode *evenH = even;
        while (odd->next && even->next) {
            odd->next = even->next;
            odd = odd->next;
            even->next =  odd->next;
            even = even->next;
        }
        odd->next = evenH;
        return head;
    }
    //389.找不同
    //每个字符异或之后 可以得到最后一个字符
    char findTheDifference(string s, string t) {
        char res = t[0];
        for (int i = 0; i < s.length(); i++) {
            res ^= s[i];
            res ^= t[i+1];
        }
        return res;
    }

    //464.我能赢吗
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger >= desiredTotal) {
            return true;
        }
        int max = maxChoosableInteger * (maxChoosableInteger + 1) / 2;
        if (max < desiredTotal) {
            return false;
        }
        unordered_map<int, bool> m;
        return canWin(maxChoosableInteger, desiredTotal, 0, m);
    }

    bool canWin(int length, int total, int used,unordered_map<int, bool> &m) {
        if (m.count(used)) {
            return m[used];
        }
        for (int i = 0; i < length; ++i) {
            int cur = (1 << i);
            if ((cur & used) == 0) {
                if (total <= i + 1 || !canWin(length, total - (i + 1), cur | used, m)) {
                    m[used] = true;
                    return true;
                }
            }
        }
        m[used] = false;
        return false;
    }

    //486.预测赢家
    bool PredictTheWinner(vector<int>& nums) {
        int n = (int)nums.size();
        if (n % 2 == 0) {
            return true;
        } else {
            vector<vector<int>> dp(n, vector<int>(n, 0));
            for (int i = 0; i < n; i++) {
                dp[i][i] = nums[i];
            }
            for (int len = 1; len < n; len++) {
                for (int i = 0, j = len; j < n; i++,j++) {
                    dp[i][j] = max(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1]);
                }
            }
            return dp[0][n-1] >= 0;
        }
    }

    //492.构造矩形
    vector<int> constructRectangle(int area) {
        int a = sqrt(area);
        vector<int> ret;
        ret.push_back(area);
        ret.push_back(1);
        int diff = area - 1;
        for (int i = 2; i <= a; i++) {
            if (area % i == 0) {
                if (abs(area / i - i) < diff) {
                    ret.clear();
                    ret.push_back(area / i);
                    ret.push_back(i);
                    diff = area / i - i;
                }
            } else {
                continue;
            }
        }
        return ret;
    }

    //500.键盘行
    vector<string> findWords(vector<string>& words) {
        map<char,int> maps = {{'q',1},{'w',1},{'e',1},{'r',1},{'t',1},{'y',1},{'u',1},{'i',1},{'o',1},{'p',1},{'a',2},{'s',2},{'d',2},{'f',2},{'g',2},{'h',2},{'j',2},{'k',2},{'l',2},{'z',3},{'x',3},{'c',3},{'v',3},{'b',3},{'n',3},{'m',3}};
        vector<string> ret;
        for (int i = 0; i<words.size(); i++) {
            string str = words[i];
            transform(str.begin(), str.end(), str.begin(), ::tolower);
            int val = maps[str[0]];
            bool vaild = true;
            for(int j = 0; j < str.length(); j++) {
                if (val != maps[str[j]]) {
                    vaild = false;
                    break;
                }
            }
            if (vaild == true) {
                ret.push_back(words[i]);
            }
        }
        return ret;
    }

    //504.七进制数
    string convertToBase7(int num) {
        int val = 0;
        int tmp = 10;
        int n = abs(num);
        do {
            val += (n % 7) * tmp / 10;
            n /= 7;
            if (n < 7) {
                val += tmp * n;
            }
            tmp *= 10;
        } while (n > 6);
        if (num < 0) {
            val = -val;
        }
        return to_string(val);
    }

    //509.斐波那契数
    int fib(int N) {
        if (N < 2) {
            return N;
        } else {
            return fib(N-1) + fib(N-2);
        }
    }

    ///下面可以保证每一个n只计算一次。
    int fib2(int N) {
        if (N < 2) {
            return N;
        } else {
            vector<int> ret;
            ret.push_back(0);
            ret.push_back(1);
            int i = 2;
            while (i <= N) {
                ret.push_back(ret[i-1] + ret[i-2]);
                i++;
            }
            return ret.back();
        }
    }

    //520. 检测大写字母
    //判断第二个字符开始和后面是否都一样大小写，如果第一个小写，判断所有是否为小写。
    bool detectCapitalUse(string word) {
        int length = (int)word.length();
        if (length < 2) {
            return true;
        }
        bool first = word[0] <= 'Z' && word[0] >= 'A';
        bool ret = word[1] <= 'Z' && word[1] >= 'A';
        for (int i = 1; i < length; i++) {
            bool cur = word[i] <= 'Z' && word[i] >= 'A';
            if (first == false && cur != false) {
                return false;
            }
            if (ret != cur) {
                return false;
            }
        }
        return true;
    }

    //541.反转字符串二
    string reverseStr(string s, int k) {
        string result = "";
        int n = (int)s.length() / (2 * k);
        if (s.length() % (2 * k) != 0) {
            n++;
        }
        for (int i = 0; i < n; i++) {
            if (i == n-1) {
                int res = (int)s.length() - 2 * i * k;
                if (res < k) {
                    result += reverse(s.substr(2 * i * k,res));
                } else if (res >= k && res <= 2 * k) {
                    result += reverse(s.substr(2 * i * k,k));
                    result += s.substr(2 * i * k + k,k);
                }
            } else {
                string sub1 = s.substr(2 * i * k,k);
                if (sub1.length() < k) {
                    result += sub1;
                } else {
                    result += reverse(sub1);
                }
                string sub2 = s.substr(2 * i * k + k,k);
                result += sub2;
            }
        }
        return result;
    }

    string reverse(string str) {
        string ret = "";
        for (int i = (int)str.length() - 1; i >= 0; i--) {
            ret += str[i];
        }
        return ret;
    }

    //554.砖墙
    int leastBricks(vector<vector<int>>& wall) {
        map<int, int> a;
        int maxnum = 0;
        for (int i = 0; i < wall.size(); i++) {
            int val = 0;
            for (int j = 0; j < wall[i].size() - 1; j++) {
                val += wall[i][j];
                ++a[val];
            }
        }
        for(map<int,int>::iterator iter = a.begin(); iter != a.end(); iter++) {
            if (iter->second == maxnum) {
                continue;
            }
            maxnum = max(maxnum, iter->second);
        }
        return (int)wall.size() - maxnum;
    }

    //561.数组拆分
    int arrayPairSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < nums.size(); i += 2) {
            res += nums[i];
        }
        return res;
    }

    //637.二叉树的层平均值
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> v;
        if (root == NULL) {
            return v;
        }
        queue<TreeNode *> parentS;
        parentS.push(root);
        queue<TreeNode *> childS;
        while (!parentS.empty() || !childS.empty()) {
            int count = int(parentS.size());
            double sum = 0;
            while (!parentS.empty()) {
                root = parentS.front();
                sum += root->val;
                if (root -> left != NULL) childS.push(root->left);
                if (root -> right != NULL) childS.push(root->right);
                parentS.pop();
            }
            parentS = childS;
            childS = queue<TreeNode *>();
            double avg = sum / count;
            v.push_back(avg);
        }
        return v;
    }

    //669.修剪二叉搜索树
    TreeNode* trimBST(TreeNode* root, int L, int R) {
        if (root == nullptr) {
            return root;
        }
        if (root->val < L) {
            return trimBST(root->right, L, R);
        }
        if (root->val > R) {
            return trimBST(root->left, L, R);
        }
        root->left = trimBST(root->left, L, R);
        root->right = trimBST(root->right, L, R);
        return root;
    }

    //693.交替二进制数
    bool hasAlternatingBits(int n) {
        bool last = (n % 2 != 0);
        while (n >>= 1) {
            if ((n % 2 != 0) == last) {
                return false;
            }
            last = !last;
        }
        return true;
    }

    //698.划分为k个相等的子集
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
        }
        int num = sum / k;
        if (sum % k != 0) {
            return false;
        }
        sort(nums.begin(), nums.end());
        if (nums.back() > num) {
            //最大值大于num了
            return false;
        }
        while (nums.size() && nums.back() == num) {
            nums.pop_back();
            k--;
        }
        vector<int> bucket(k,0);
        return partitionKSubsets(nums, bucket, num);
    }

    bool partitionKSubsets(vector<int>& nums, vector<int>& bucket, int num) {
        if (nums.size() == 0) {
            return true;
        }
        int v = nums.back();
        nums.pop_back();
        for (auto it = bucket.begin(); it != bucket.end(); it++) {
            if (v + *it <= num) {
                *it += v;
                if (partitionKSubsets(nums, bucket, num)) {
                    return true;
                }
                *it -= v;
            }
            if (*it == 0) {
                break;
            }
        }
        return false;
    }

    //725.分隔链表
    vector<ListNode*> splitListToParts(ListNode* root, int k) {
        ListNode *p = root;
        int i = 0;
        while (p) {
            i++;
            p = p->next;
        }
        int count = i / k;
        int remainder = i % k;
        vector<ListNode *> res;
        for (int i = 0; i < k; i++) {
            ListNode *newH = new ListNode(0);
            newH->next = root;
            ListNode *cur = newH;
            int newC = count;
            if (remainder > 0) {
                newC++;
                remainder--;
            }
            while (newC > 0) {
                root = root->next;
                cur = cur->next;
                newC--;
            }
            cur->next = NULL;
            res.push_back(newH->next);
        }
        return res;
    }

    //817.链表组件
    int numComponents(ListNode* head, vector<int>& G) {
        unordered_set<int> set(G.begin(),G.end());
        int res = 0;
        while (head) {
            if (set.find(head->val) != set.end()) {
                if (!head->next || set.find(head->next->val) == set.end()) {
                    res++;
                }
            }
            head = head->next;
        }
        return res;
    }

    //876.删除链表中间节点
    //快慢指针方法：当快指针走完 慢指针就是中间节点
    ListNode* middleNode(ListNode* head) {
        ListNode *low = head;
        ListNode *fast = head;
        while (fast && fast->next) {
            low = low->next;
            fast = fast->next->next;
        }
        return low;
    }

    //912.排序数组
    vector<int> sortArray(vector<int>& nums) {
//        for (int i = 0; i < nums.size(); i++) {
//            for (int j = i; j < nums.size(); j++) {
//                if (nums[i] > nums[j]) {
//                    int temp = nums[i];
//                    nums[i] = nums[j];
//                    nums[j] = temp;
//                }
//            }
//        }
//        return nums;
        sortArr(nums,0,nums.size() - 1);
        return nums;
    }

    void sortArr(vector<int> &nums, int left, int right) {
        if (left >= right) {
            return;
        }
        int index = quickSort(nums, left, right);
        sortArr(nums, left, index - 1);
        sortArr(nums, index + 1, right);
    }

    int quickSort(vector<int> &nums, int left, int right) {
        int temp = nums[left];
        while (left < right) {
            while (left < right && nums[right] > temp) {
                right--;
            }
            nums[left] = nums[right];
            while (left < right && nums[left] < temp) {
                left++;
            }
            nums[right] = nums[left];
        }
        nums[left] = temp;
        return left;
    }

    //914.卡牌分组
    bool hasGroupsSizeX(vector<int>& deck) {
        map<int,int> map;
        for (int i = 0; i < deck.size(); i++) {
            map[deck[i]]++;
        }
        int res = map.begin()->second;
        for(auto i = map.begin(); i != map.end(); i++) {
            res = gcd(res, i->second);
        }
        return res >= 2;
    }

    int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a%b);
    }

    //1033.移动石子直至连续
    vector<int> numMovesStones(int a, int b, int c) {
        if (a>b) {
            swap(a, b);
        }
        if (b>c) {
            swap(b, c);
        }
        if (a>b) {
            swap(a, b);
        }
        if (b-a==1 && c-b==1) {
            return vector<int>{0,0};
        }
        if (b-a<=2 || c-b<=2) {
            return vector<int>{1,c-a-2};
        }
        return vector<int>{2,c-a-2};
    }

    //1137.第n个斐波那契数
    int tribonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        vector<int> res = vector<int>(n+1,0);
        res[0] = 0;
        res[1] = 1;
        res[2] = 1;
        for (int i = 3; i <= n; i++) {
            res[i] = res[i-1] + res[i-2] + res[i-3];
        }
        return res[n];
    }

    //1171.从链表中删去总和值为零的连续节点
    ListNode* removeZeroSumSublists(ListNode* head) {
        ListNode *cur = new ListNode(0);
        ListNode *p = cur;
        cur->next = head;
        while (cur) {
            ListNode *newH = cur->next;
            int tempSum = 0;
            while (newH) {
                tempSum += newH->val;
                newH = newH->next;
                if (tempSum == 0) {
                    cur->next = newH;
                    break;
                }
            }
            if (newH == NULL) {
                cur = cur->next;
            }
        }
        return p->next;
    }

    //1290.二进制链表转整数
    int getDecimalValue(ListNode* head) {
        int res = 0;
        while (head) {
            res *= 2;
            res += head->val;
            head = head->next;
        }
        return res;
    }


    //每日一题4.7
    void rotate111(vector<vector<int>>& matrix) {
        if (matrix.size() < 2) {
            return;
        }
        unsigned long size = matrix.size();
        for (int i = 0; i < size / 2; i++) {
            for (int j = i; j < size - i - 1; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[size-j-1][i];
                matrix[size-j-1][i] = matrix[size-i-1][size-j-1];
                matrix[size-i-1][size-j-1] = matrix[j][size-i-1];
                matrix[j][size-i-1] = tmp;
            }
        }
    }

    //面试题34.二叉树中和为某一值的路径
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        if (!root) {
            return res;
        }
        vector<int> tmp{root->val};
        bactracePathSum(res, sum, root->val, tmp, root);
        return res;
    }

    void bactracePathSum(vector<vector<int>>& res, int sum, int currentSum, vector<int>& tmp, TreeNode *root) {
        if (currentSum == sum && !root->left && !root->right) {
            res.push_back(tmp);
            return;
        }
        if (root->left) {
            tmp.push_back(root->left->val);
            bactracePathSum(res, sum, currentSum + root->left->val, tmp, root->left);
            tmp.pop_back();
        }
        if (root->right) {
            tmp.push_back(root->right->val);
            bactracePathSum(res, sum, currentSum + root->right->val, tmp, root->right);
            tmp.pop_back();
        }
    }
    //面试题02.01
    //使用set保存之前存在的节点 时间复杂度O(n),空间复杂的O(n)
    //不使用额外空间的话就是2层循环 删除后面的链表节点 时间复杂度O(n*n)
    ListNode* removeDuplicateNodes(ListNode* head) {
        if (!head) {
            return NULL;
        }
        ListNode *p = head;
        unordered_set<int> set;
        set.insert(head->val);
        while (p->next) {
            if (set.find(p->next->val) != set.end()) {
                p->next = p->next->next;
            } else {
                set.insert(p->next->val);
                p = p->next;
            }
        }
        return head;
    }

    //面试题03.数组中的重复数字
    int findRepeatNumber(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            while (i != nums[i]) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                int tmp = nums[i];
                nums[i] = nums[tmp];
                nums[tmp] = tmp;
            }
        }
        return -1;
    }

    //面试题04. 二维数组中的查找
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.size() < 1) {
            return false;
        }
        int row = 0;
        int col = matrix[0].size()-1;
        while (row < matrix.size() && col >= 0) {
            int num = matrix[row][col];
            if (num == target) {
                return true;
            } else if (num > target) {
                col--;
            } else {
                row++;
            }
        }
        return false;
    }

    //面试题04.10. 检查子数
    bool checkSubTree(TreeNode* t1, TreeNode* t2) {
        if (!t1 || !t2) {
            return false;
        }
        return dfsCheck(t1, t2) || checkSubTree(t1->left, t2) || checkSubTree(t1->right, t2);
    }

    bool dfsCheck(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2) {
            return true;
        }
        if (t1->val != t2->val) {
            return false;
        }
        return dfsCheck(t1->left, t2->left) && dfsCheck(t1->right, t2->right);
    }

    bool queryString(string S, int N) {
        for (int i = 1; i <= N; i++) {
            if (S.find(to_bin(N)) == -1) {
                return false;
            }
        }
        return true;
    }

    string to_bin( int N){
        string res = "";
        while(N){
            res = char( N % 2 + '0') + res;
            N/=2;
        }
        return res;
    }


    ListNode *detectCycle2(ListNode *head) {
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                slow = head;
                while (slow != fast) {
                    slow = slow->next;
                    fast = fast->next;
                }
                return slow;
            }
        }
        return NULL;
    }

    //面试题07.重建二叉树
    TreeNode* buildTree22(vector<int>& preorder, vector<int>& inorder) {
        int preLeft = 0;
        int preRight = preorder.size() - 1;
        int inLeft = 0;
        int inRight = inorder.size() - 1;
        return buildTreeFun22(preorder, inorder, preLeft, preRight, inLeft, inRight);
    }

    TreeNode *buildTreeFun22(vector<int> &preprder, vector<int>& inorder, int preL, int preR, int inL, int inR) {
        if (preL > preR || inL > inR) {
            return NULL;
        }
        TreeNode *node = new TreeNode(preprder[preL]);
        int index = inL;
        for (int i = 0; i < inorder.size(); i++) {
            if (inorder[i] == preprder[preL]) {
                index = i;
            }
        }
        int left = index - inL;
        node->left = buildTreeFun22(preprder, inorder, preL + 1, preL + left, inL, index-1);
        node->right = buildTreeFun22(preprder, inorder, preL + left + 1, preR, index + 1, inR);
        return node;
    }

    //面试题10- I. 斐波那契数列
    int fib3(int n) {
        if (n <= 1) {
            return n;
        }
        vector<int> dp(n+1);
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n] % 1000000007;
    }

    //面试题10- II. 青蛙跳台阶问题
    int numWays(int n) {
        if (n <= 1) {
            return 1;
        }
        int a = 1;
        int b = 1;
        int c = 0;
        for (int i = 2; i <= n; i++) {
            c = (a + b) % 1000000007;
            a = b;
            b = c;
        }
        return c;
    }

    //面试题11. 旋转数组的最小数字
    int minArray(vector<int>& numbers) {
        int left = 0;
        int right = numbers.size() - 1;
        while (left < right) {
            int mid = left + (right - left)/2;
            if (numbers[mid] > numbers[right]) {
                left = mid + 1;
            } else if (numbers[mid] < numbers[right]){
                right = mid;
            } else {
                right--;
            }
        }
        return numbers[left];
    }

    //面试题12. 矩阵中的路径
    bool exist(vector<vector<char>>& board, string word) {
        if (word.size() < 1 || board.size() < 1) {
            return false;
        }
        vector<vector<bool>> routs(board.size(),vector<bool>(board[0].size(),false));
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (board[i][j] == word[0]) {
                    if (backTrackExist(routs, board, i, j, word, 0)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool backTrackExist(vector<vector<bool>>& routs, vector<vector<char>>& board, int row, int col, string word, int i) {
        if (i == word.size()) {
            return true;
        }
        if (row < 0 || row >= board.size() || col < 0 || col >= board[0].size() || routs[row][col] || board[row][col] != word[i]) {
            return false;
        }
        routs[row][col] = true;
        bool res = backTrackExist(routs, board, row + 1, col, word, i+1) || backTrackExist(routs, board, row - 1, col, word, i+1) || backTrackExist(routs, board, row, col + 1, word, i+1) || backTrackExist(routs, board, row, col - 1, word, i+1);
        routs[row][col] = false;
        return res;
    }

    //面试题13. 机器人的运动范围
    //有些地方是不可到达的 成等腰三角形
    int movingCount(int m, int n, int k) {
        vector<vector<bool>> visited(m,vector<bool>(n,false));
        return dfsMovingCount(0, 0, m, n, k, visited);
    }

    int dfsMovingCount(int i, int j, int m, int n, int k, vector<vector<bool>>& visited) {
        if (i >= m || j >= n || visited[i][j] || sumRowCol(i, j) > k) {
            return 0;
        }
        visited[i][j] = true;
        return 1 + dfsMovingCount(i+1, j, m, n, k, visited) + dfsMovingCount(i, j+1, m, n, k, visited);
    }

    int sumRowCol(int m, int n) {
        int sum = 0;
        while (m > 0) {
            sum += m % 10;
            m /= 10;
        }
        while (n > 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }

    //面试题14-1 剪绳子
    int cuttingRope(int n) {
        if (n == 2) {
            return 1;
        }
        vector<int> dp(n+1);
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                dp[i] = max(dp[i], max((i-j)*j, j*dp[i-j]));
            }
        }
        return dp[n];
    }

    //链表头插法 尾插法
    ListNode *headCreateListNode(vector<int> &nums) {
        ListNode *head = NULL;
        for (int i = 0; i < nums.size(); i++) {
            ListNode *newNode = new ListNode(nums[i]);
            newNode->next = head;
            head = newNode;
        }
        return head;
    }

    ListNode *tailCreateListNode(vector<int> &nums) {
        ListNode *head = new ListNode(0);
        ListNode *p = head;
        for (int i = 0; i < nums.size(); i++) {
            ListNode *newNode = new ListNode(nums[i]);
            p->next = newNode;
            p = p->next;
        }
        return head->next;
    }

    //面试题21. 调整数组顺序使奇数位于偶数前面
    vector<int> exchange(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        while (left < right) {
            while (right >= 0 && nums[right] % 2 == 0) {
                right--;
            }
            while (left <= nums.size()-1 && nums[left] % 2 == 1) {
                left++;
            }
            if (left >= right) {
                break;
            }
            int tmp = nums[right];
            nums[right] = nums[left];
            nums[left] = tmp;
        }
        return nums;
    }

    //面试题22.链表的倒数第k个节点
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *fast = head;
        ListNode *slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast->next;
        }
        while (fast) {
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }

    //面试题24.反转链表
    ListNode* reverseList2(ListNode* head) {
        ListNode *newH = NULL;
        for (ListNode *p = head; p; ) {
            ListNode *temp = p->next;
            p->next = newH;
            newH = p;
            p = temp;
        }
        return newH;
    }

    //面试题25.合并两个排序链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode *newH = new ListNode(0);
        ListNode *p = newH;
        while (l1 || l2) {
            if (!l1) {
                p->next = l2;
                break;
            }
            if (!l2) {
                p->next = l1;
                break;
            }
            if (l1->val <= l2->val) {
                ListNode *tmp = l1;
                l1 = l1->next;
                tmp->next = NULL;
                p->next = tmp;
                p = p->next;
            } else {
                ListNode *tmp = l2;
                l2 = l2->next;
                tmp->next = NULL;
                p->next = tmp;
                p = p->next;
            }
        }
        return newH->next;
    }

    //面试题26.树的子结构
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if (!A || !B) {
            return false;
        }
        return isSubStructDFS(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }

    bool isSubStructDFS(TreeNode* A, TreeNode* B) {
        if (!B) {
            return true;
        }
        if (!A) {
            return false;
        }
        return A->val == B->val && isSubStructDFS(A->left, B->left) && isSubStructDFS(A->right, B->right);
    }

    //面试题27. 二叉树的镜像
    TreeNode* mirrorTree(TreeNode* root) {
        if (!root) {
            return NULL;
        }
        TreeNode *left = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(left);
        return root;
    }

    //面试题28. 对称的二叉树
    bool isSymmetric(TreeNode* root) {
        if (!root) {
            return true;
        }
        return helpIsSymmetric(root->left, root->right);
    }

    bool helpIsSymmetric(TreeNode *left, TreeNode *right) {
        if (!left && !right) {
            return true;
        }
        if (!left || !right) {
            return false;
        }
        return left->val == right->val && helpIsSymmetric(left->left, right->right) && helpIsSymmetric(left->right, right->left);
    }

    //面试题29. 顺时针打印矩阵
    vector<int> spiralOrder2(vector<vector<int>>& matrix) {
        vector<int> res;
        int top = 0;
        int bottom = matrix.size() - 1;
        if (bottom < 0) {
            return res;
        }
        int left = 0;
        int right = matrix[0].size() - 1;
        while (top <= bottom && left <= right) {
            for (int i = left; i <= right; i++) {
                res.push_back(matrix[top][i]);
            }
            top++;
            if (top > bottom) {
                break;
            }
            for (int j = top; j <= bottom; j++) {
                res.push_back(matrix[j][right]);
            }
            right--;
            if (right < left) {
                break;
            }
            for (int k = right; k >= left; k--) {
                res.push_back(matrix[bottom][k]);
            }
            bottom--;
            if (bottom < top) {
                break;
            }
            for (int l = bottom; l >= top; l--) {
                res.push_back(matrix[l][left]);
            }
            left++;
            if (left > right) {
                break;
            }
        }
        return res;
    }

    //面试题31.栈的压入、弹出序列
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if (pushed.size() != popped.size()) {
            return false;
        }
        stack<int> stack;
        int j = 0;
        for (int i = 0; i < pushed.size();i++) {
            stack.push(pushed[i]);
            while (j < popped.size() && !stack.empty() && stack.top() == popped[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.empty();
    }

    //面试题32 - I. 从上到下打印二叉树
    vector<int> levelOrderTop(TreeNode* root) {
        vector<int> res;
        if (!root) {
            return res;
        }
        queue<TreeNode *> qqq;
        qqq.push(root);
        while (!qqq.empty()) {
            TreeNode *node = qqq.front();
            res.push_back(node->val);
            if (node->left) {
                qqq.push(node->left);
            }
            if (node->right) {
                qqq.push(node->right);
            }
            qqq.pop();
        }
        return res;

//        vector<int> res;
//        if (!root) {
//            return res;
//        }
//        queue<TreeNode *> parents;
//        queue<TreeNode *> childs;
//        parents.push(root);
//        while (!parents.empty() || !childs.empty()) {
//            while (!parents.empty()) {
//                TreeNode *node = parents.front();
//                res.push_back(node->val);
//                if (node->left) {
//                    childs.push(node->left);
//                }
//                if (node->right) {
//                    childs.push(node->right);
//                }
//                parents.pop();
//            }
//            parents = childs;
//            childs = queue<TreeNode *>();
//        }
//        return res;
    }

    //面试题32 - II. 从上到下打印二叉树 II
    vector<vector<int>> levelOrderTop2(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) {
            return res;
        }
        queue<TreeNode *> parents;
        queue<TreeNode *> childs;
        parents.push(root);
        while (!parents.empty() || !childs.empty()) {
            vector<int> tmp;
            while (!parents.empty()) {
                TreeNode *node = parents.front();
                tmp.push_back(node->val);
                if (node->left) {
                    childs.push(node->left);
                }
                if (node->right) {
                    childs.push(node->right);
                }
                parents.pop();
            }
            parents = childs;
            childs = queue<TreeNode *>();
            res.push_back(tmp);
        }
        return res;
    }

    //面试题33. 二叉搜索树的后序遍历序列
    bool verifyPostorder(vector<int>& postorder) {
        if (postorder.size() < 2) {
            return true;
        }
        bool a = verifyPostorderIndex(postorder, 0, (int)postorder.size() - 1);
        return a;
    }

    bool verifyPostorderIndex(vector<int>& postorder,int left,int right) {
        if (left >= right) {
            return true;
        }
        int root = postorder[right];
        int mid = left-1;
        while(mid < right && postorder[mid] < root) {
            mid++;
        }
        for (int i = mid; i < right;i++) {
            if (postorder[i] < root) {
                return false;
            }
        }
        return verifyPostorderIndex(postorder, left, mid-1) && verifyPostorderIndex(postorder, mid, right-1);
    }

    //面试题38. 字符串的排列
    vector<string> permutation(string s) {
        vector<string> res;
        vector<bool> visited(s.size()-1,false);
        sort(s.begin(), s.end());
        dfsPermutation(s, "", visited, res);
        return res;
    }

    void dfsPermutation(string s,string p,vector<bool>& visited, vector<string>& res) {
        if (p.size() == s.size()) {
            res.push_back(p);
            return;
        }
        for (int i = 0; i < s.size(); i++) {
            if (visited[i]) {
                continue;
            }
            if (i > 0 && !visited[i-1] && s[i-1] == s[i]) {
                continue;
            }
            p.push_back(s[i]);
            visited[i] = true;
            dfsPermutation(s, p, visited, res);
            p.pop_back();
            visited[i] = false;
        }
    }

    //面试题39. 数组中出现次数超过一半的数字
    //摩尔投票法
    int majorityElement(vector<int>& nums) {
        int res = nums[0];
        int count = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (res == nums[i]) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    res = nums[i];
                    count++;
                }
            }
        }
        return  res;
    }

    //面试题56 - I. 数组中数字出现的次数
    //全部^一次就得到a和b ^ 的值，然后判断从右往左第一个不相同的位记为h，再把数组氛围2组，分别^最终得到a和b
    vector<int> singleNumbers(vector<int>& nums) {
        int a = 0;
        int b = 0;
        int c = 0;
        for (auto i : nums) {
            c ^= i;
        }
        int h = 1;
        while ((c&h) == 0) {
            h <<= 1;
        }
        for (auto i : nums) {
            if ((i & h) == 0) {
                a ^= i;
            } else {
                b ^= i;
            }
        }
        return vector<int>{a,b};
    }

    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        for (int i = 0; i < nums.size(); i++) {
            int a = nums[i];
            int left = i;
            int right = nums.size() - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (nums[mid] == target - a) {
                    return vector<int>{a, target-a};
                } else if (nums[mid] < target - a) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return res;
    }

    //面试题57 - II. 和为s的连续正数序列
    vector<vector<int>> findContinuousSequence(int target) {
        int len = sqrt(2 * target) + 1;
        vector<vector<int>> res;
        for (int i = len; i >= 2; i--) {
            if (2 * target % i == 0) {
                int tmp = 2 * target / i - i + 1;
                if (tmp > 0 && tmp % 2 == 0) {
                    int a1 = tmp / 2;
                    vector<int> subRes;
                    for (int j = a1; j < a1+i; j++) {
                        subRes.push_back(j);
                    }
                    res.push_back(subRes);
                }
            }
        }
        return res;
    }

    //面试题63.股票的最大利润
    int maxProfit(vector<int>& prices) {
        if (prices.size() < 2) {
            return 0;
        }
        int res = 0;
        int min = prices[0];
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else {
                res = max(res, prices[i] - min);
            }
        }
        return res;
    }
};

