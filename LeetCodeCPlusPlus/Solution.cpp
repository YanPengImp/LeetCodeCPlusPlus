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

    //4. 寻找两个正序数组的中位数
    // m+n+1)/2 m+n+2)/2找出这两个数就不用区分总个数的奇偶
    //就相当于是两数组中寻找第k大元素，先分别在数组中寻找第k/2个数 比较，如果某一个的较小 则舍弃前半部分。
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = (int)nums1.size();
        int n = (int)nums2.size();
        int left = (m+n+1)/2;
        int right = (m+n+2)/2;
        return (findKthInTwoNums(nums1, 0, nums2, 0, left) + findKthInTwoNums(nums1, 0, nums2, 0, right)) / 2.0;
    }

    int findKthInTwoNums(vector<int>& nums1, int l, vector<int>& nums2, int r, int k) {
        if (l >= nums1.size()) {
            return nums2[r+k-1];
        }
        if (r >= nums2.size()) {
            return nums1[l+k-1];
        }
        if (k==1) {
            return min(nums1[l],nums2[r]);
        }
        int mid1 = l + k/2 - 1 < nums1.size() ? nums1[l+k/2-1] : INT_MAX;
        int mid2 = r + k/2 - 1 < nums2.size() ? nums2[r+k/2-1] :INT_MAX;
        if (mid1 < mid2) {
            return findKthInTwoNums(nums1, l + k/2, nums2, r, k-k/2);
        } else {
            return findKthInTwoNums(nums1, l, nums2, r+k/2, k-k/2);
        }
    }

    //5. 最长回文子串
    string longestPalindrome(string s) {
        string res = "";
        for (int i = 0; i < s.size(); i++) {
            string s1 = longestPalindromeSearch(s, i, i);
            string s2 = longestPalindromeSearch(s, i, i+1);
            res = s1.size() > res.size() ? s1 : res;
            res = s2.size() > res.size() ? s2 : res;
        }
        return res;
    }

    string longestPalindromeSearch(string s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            left--;
            right++;
        }
        return s.substr(left+1, right-left+1);
    }

    //9.回文数
    bool isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int res = 0;
        int t = x;
        while (t) {
            res = res * 10 + t % 10;
            t /= 10;
        }
        return res == x;
    }

    //10.正则表达式匹配
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();
        if (m == 0 && n == 0) {
            return true;
        }
        if (m == 0 || n == 0) {
            return false;
        }
        vector<vector<bool>> dp(m+1,vector<bool>(n+1,false));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (i >= 2 && p[i-1] == '*' && p[i-2]) {
                dp[0][i] = dp[0][i-2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i-1] == p[j-1] || p[j-1] == '.') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p[j-1] == '*') {
                    if (p[j-2] != s[i-1] && p[j-2] != '.') {
                        dp[i][j] = dp[i][j-2];
                    } else {
                        dp[i][j] = dp[i-1][j] || dp[i][j-2] || dp[i][j-1];
                    }
                }
            }
        }
        return dp[m][n];
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

    //14.最长公共前缀
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) {
            return "";
        }
        string res = "";
        for (int i = 0; i < strs[0].size(); i++) {
            for (int j = 1; j < strs.size(); j++) {
                if (i == strs[j].size() || strs[0][i] != strs[j][i]) {
                    return res;
                }
            }
            res += strs[0][i];
        }
        return res;
    }

    //15. 三数之和
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for (int first = 0; first < n; first++) {
            if (first > 0 && nums[first] == nums[first-1]) {
                continue;
            }
            int target = -nums[first];
            for (int second = first + 1; second < n; second++) {
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                int third = n-1;
                while (second < third && nums[second] + nums[third] > target) {
                    third--;
                }
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    res.push_back(vector<int>{nums[first], nums[second], nums[third]});
                }
            }
        }
        return res;
    }

    //16.最接近的三数之和
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int closest = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.size()-2; i++) {
            int l = i+1;
            int r = nums.size()-1;
            while (l < r) {
                int num = nums[i] + nums[l] + nums[r];
                if (abs(num - target) < abs(closest - target)) {
                    closest = num;
                }
                if (num > target) {
                    r--;
                } else if (num < target) {
                    l++;
                } else {
                    return target;
                }
            }
        }
        return closest;
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

    //32.最长有效括号
    int longestValidParentheses(string s) {
        stack<int> stack;
        stack.push(-1);
        int res = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    res = max(res, i - stack.top());
                }
            }
        }
        return res;
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

    //39.组合总数
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> tmp;
        dfsCombinationSum(candidates, res, tmp, target, 0);
        return res;
    }

    void dfsCombinationSum(vector<int>& candidates, vector<vector<int>>& res, vector<int>& tmp, int target, int index) {
        if (target == 0) {
            res.push_back(tmp);
            return;
        }
        if (target < 0) {
            return;
        }
        for (int i = index; i < candidates.size(); i++) {
            if (candidates[i] > target) {
                continue;
            }
            tmp.push_back(candidates[i]);
            dfsCombinationSum(candidates, res, tmp, target - candidates[i], i);
            tmp.pop_back();
        }
    }

    //40. 组合总和 II
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> tmp;
        sort(candidates.begin(), candidates.end());
        dfsCombinationSum2(candidates, res, tmp, target, 0);
        return res;
    }

    void dfsCombinationSum2(vector<int>& candidates, vector<vector<int>>& res, vector<int>& tmp, int target, int index) {
        if (target == 0) {
            res.push_back(tmp);
            return;
        }
        if (target < 0) {
            return;
        }
        for (int i = index; i < candidates.size(); i++) {
            if (candidates[i] > target) {
                continue;
            }
            //去重
            if (i > index && candidates[i] == candidates[i-1]) {
                continue;
            }
            tmp.push_back(candidates[i]);
            dfsCombinationSum2(candidates, res, tmp, target - candidates[i], i+1);
            tmp.pop_back();
        }
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

    int trap2(vector<int>& height) {
        if (height.size() < 1) {
            return 0;
        }
        int left = 0;
        int right = height.size()-1;
        int leftM = height[left];
        int rightM = height[right];
        int res = 0;
        while (left < right) {
            leftM = max(leftM,height[left]);
            rightM = max(rightM,height[right]);
            if (leftM < rightM) {
                res += (leftM - height[left]);
                left++;
            } else {
                res += (rightM - height[right]);
                right--;
            }
        }
        return res;
    }

    //44. 通配符匹配
    bool isMatch22(string s, string p) {
        int m = (int)s.size();
        int n = (int)p.size();
        vector<vector<bool>> dp(m+1,vector<bool>(n+1,false));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (p[i-1] == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p[j-1] == '*') {
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
                } else if (s[i-1] == p[j-1] || p[j-1] == '?') {
                    dp[i][j] = dp[i-1][j-1];
                }
            }
        }
        return dp[m][n];
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

    //47.全排列II
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        allPermuteUnique(res, nums, 0);
        return res;
    }

    //v传值 不传引用 swap不用交换回来
    void allPermuteUnique(vector<vector<int>> &res, vector<int> v, int start) {
        if (start == v.size() - 1) {
            res.push_back(v);
            return;
        }
        for (int i = start; i < v.size(); i++) {
            if (i > start && v[i] == v[start]) {
                continue;
            }
            int t = v[start];
            v[start] = v[i];
            v[i] = t;
            allPermuteUnique(res, v, start+1);
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

    vector<vector<int>> merge2(vector<vector<int>>& intervals) {
        int n = (int)intervals.size();
        vector<int> starts, ends;
        vector<vector<int>> res;
        for (int i = 0; i < n; i++) {
            starts.push_back(intervals[i][0]);
            ends.push_back(intervals[i][1]);
        }
        sort(starts.begin(), starts.end());
        sort(ends.begin(), ends.end());
        for (int i = 0, j = 0; i < n; i++) {
            if (i == n - 1 || starts[i+1] > ends[i]) {
                res.push_back(vector<int>{starts[j], ends[i]});
                j = i+1;
            }
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

    //63.不同路径II
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
//        dfsUniquePathsWithObstacles(obstacleGrid, 0, 0, count);
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m+1, vector<int>(n+1,0));
        dp[0][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (obstacleGrid[i-1][j-1] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m][n];
    }

    //dfs超时 还是用dp吧
    void dfsUniquePathsWithObstacles(vector<vector<int>>& obstacleGrid, int row, int col, int& count) {
        if (row >= obstacleGrid.size() || col >= obstacleGrid[0].size()) {
            return;
        }
        if (obstacleGrid[row][col] == 1) {
            return;
        }
        if (row == obstacleGrid.size() - 1 && col == obstacleGrid[0].size() - 1) {
            count += 1;
            return;
        }
        dfsUniquePathsWithObstacles(obstacleGrid, row + 1, col, count);
        dfsUniquePathsWithObstacles(obstacleGrid, row, col + 1, count);
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

    //69. x 的平方根
    int mySqrt(int x) {
        if (x <= 1) {
            return 1;
        }
        int left = 0;
        int right = x;
        int res = 0;
        while (left < right) {
            res = (right + left)/2;
            if (res * res == x) {
                return res;
            }
            if (res * res == x || (res * res < x && (res+1)*(res+1) > x)) {
                return res;
            } else if (res * res < x) {
                left = res;
            } else {
                right = res;
            }
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

    //72.编辑距离
    //s1 通过改变次数转为s2
    //dp[i][j]表示s1[0~i-1]转到s2[0~j-1]所需步数
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i-1] == word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    int minN = min(dp[i][j-1],dp[i-1][j]);
                    minN = min(minN,dp[i-1][j-1]);
                    dp[i][j] = minN + 1;
                }
            }
        }
        return dp[m][n];
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

    //87. 扰乱字符串
    //递归法
    bool isScramble(string s1, string s2) {
        if (s1.size() != s2.size()) {
            return false;
        }
        if (s1 == s2) {
            return true;
        }
        string t1 = s1;
        string t2 = s2;
        sort(t1.begin(), t1.end());
        sort(t2.begin(), t2.end());
        if (t1 != t2) {
            return false;
        }
        for (int i = 1; i < s1.size(); i++) {
            bool flag1 = isScramble(s1.substr(0,i), s2.substr(0,i)) && isScramble(s1.substr(i,s1.size()-i), s2.substr(i,s2.size()-i));
            bool flag2 = isScramble(s1.substr(0,i), s2.substr(s2.size()-i,i)) && isScramble(s1.substr(i,s1.size()-i), s2.substr(0,s2.size()-i));
            if (flag1 || flag2) {
                return true;
            }
        }
        return false;
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

    //93. 复原IP地址
    //leetcode编译不过 很烦。
    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        string tmp;
        dfsRestoreIpAddresses(res, s, tmp, 0);
        return res;
    }

    //ip 每一段必为1-3位，判断这一段是否符合要求然后组成一个数字 遍历剩余的s组成3个数字。。。依次到最后组成4个数字 并且s刚好遍历完
    void dfsRestoreIpAddresses(vector<string>& res, string s, string& tmp, int num) {

        if (num >= 4) {
            if (s.empty()) {
                res.push_back(tmp);
            }
            return;
        }
        if (num > 0) {
            tmp += '.';
        }
        for (int i = 1; i <= 3 && i <= s.size(); i++) {
            if (validIpString(s.substr(0,i))) {
                tmp += s.substr(0,i);
                dfsRestoreIpAddresses(res, s.substr(i,s.size()-i), tmp, num+1);
                tmp.erase(tmp.size()-i,i);
            }
        }
        tmp.pop_back();
    }

    bool validIpString(string s) {
        if (s.empty() || (s[0] == '0' && s.size() > 1)) {
            return false;
        }
        int i = stoi(s);
        if (i >= 0 && i <= 255) {
            return true;
        }
        return false;
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

    ///111.二叉树最小深度
    int minDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        int left = minDepth(root->left);
        int right = minDepth(root->right);
        if (left == 0 || right == 0) {
            return left + right + 1;
        }
        return min(left, right) + 1;
    }

    //112. 路径总和
    bool hasPathSum(TreeNode* root, int sum) {
        if (!root) {
            return false;
        }
        if (!root->left && !root->right) {
            return root->val == sum;
        }
        return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
    }

    //115.不同的子序列
    /*
     *    *  b  a  b  g  b  a  g
     * *  1  1  1  1  1  1  1  1
     * b  0  1  1  2  2  3  3  3
     * a  0  0  1  1  1  1  4  4
     * g  0  0  0  0  1  1  1  5
     */
    int numDistinct(string s, string t) {
        int m = t.size();
        int n = s.size();
        vector<vector<long>> dp(m+1, vector<long>(n+1,0));
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (t[i-1] == s[j-1]) {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                } else {
                    dp[i][j] = dp[i][j-1];
                }
            }
        }
        return dp[m][n];
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

    int minimumTotal2(vector<vector<int>>& triangle) {
        for (int i = triangle.size() - 2; i >= 0; i--) {
            for (int j = 0; j <= triangle[i].size() - 1; j++) {
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1]);
            }
        }
        return triangle[0][0];
    }

    //124. 二叉树中的最大路径和
    int sum = INT_MIN;
    int maxPathSum(TreeNode* root) {
        dfsMaxPathSum(root);
        return sum;
    }

    int dfsMaxPathSum(TreeNode *root) {
        if (!root) {
            return 0;
        }
        int left = max(0, dfsMaxPathSum(root->left));
        int right = max(0, dfsMaxPathSum(root->right));
        sum = max(sum, root->val + left + right);
        return max(left, right) + root->val;
    }

    //125.验证回文串 只考虑字母和数字 不考虑大小写
    bool isPalindrome(string s) {
        if (s.size() == 0) {
            return true;
        }
        int l = 0;
        int r = s.size()-1;
        while (l <= r) {
            char lc = s[l];
            char lr = s[r];
            if (isalnum(lc) && isalnum(lr)) {
                if (lc > 96) {
                    lc -= 32;
                }
                if (lr > 96) {
                    lr -= 32;
                }
                if (lc == lr) {
                    l++;
                    r--;
                } else {
                    return false;
                }
            } else {
                if(isalnum(lc) == 0) {
                    l++;
                }
                if(isalnum(lr) == 0) {
                    r--;
                }
            }
        }
        return true;
    }

    vector<int> newFib(int n) {
        vector<int> dp(n);
        dp[0] = 1;
        int sum = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = 2 * sum;
            sum += dp[i];
        }
        return dp;
    }

    //126.单词接龙
    //最短路径应该需要用到bfs，这儿用的dfs终止条件怎么判断不符合条件的呢。
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        vector<vector<int>> transform = vector<vector<int>>(wordList.size(), vector<int>());
        for (int i = 0; i < wordList.size(); i++) {
            for (int j = 0; j < wordList.size(); j++) {
                if (i == j) {
                    continue;
                }
                if (canTransform(wordList[i], wordList[j])) {
                    transform[i].push_back(j);
                }
            }
        }
        vector<vector<string>> res;
        int begin = 0;
        int end = 0;
        for (int i = 0; i < wordList.size(); i++) {
            if (beginWord == wordList[i]) {
                begin = i;
            }
            if (endWord == wordList[i]) {
                end = i;
            }
        }
        vector<string> tmp = vector<string>();
        dfs(res, tmp, wordList, transform, begin, end);
        return res;
    }

    void dfs(vector<vector<string>>& res, vector<string>& tmp, vector<string>& wordList, vector<vector<int>>& transform,int begin, int end) {
        if (tmp.size() > wordList.size()) {
            return;
        }
        if (begin == end) {
            res.push_back(tmp);
            return;
        }
        tmp.push_back(wordList[begin]);
        for (int i = 0; i < transform[begin].size(); i++) {
            dfs(res, tmp, wordList, transform, transform[begin][i], end);
        }
        tmp.pop_back();
    }

    bool canTransform(string s1, string s2) {
        if (s1.size() != s2.size()) {
            return false;
        }
        int i = 0;
        int count = 1;
        while (i < s1.size()) {
            if (s1[i] != s2[i]) {
                count--;
            }
        }
        return count == 0;
    }



    //128.最长连续序列
    //储存已key为边界的最长序列长度
    //比如123 56
    //map[1] = 3 map[3] = 3 map[5] = 2 map[6] = 2
    //此时扫描到4，那么会去找3和5是否存在，就是3+2+1的长度变为123456了。
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

    //147. 对链表进行插入排序
    ListNode* insertionSortList(ListNode* head) {
        ListNode *dumpHead = new ListNode(0);
        dumpHead->next = head;
        ListNode *pre = nullptr;
        while (head && head->next) {
            //这种排好的继续遍历
            if (head->val <= head->next->val) {
                head = head->next;
                continue;
            }
            pre = dumpHead;
            while (pre->next->val < head->next->val) {
                pre = pre->next;
            }
            ListNode *cur = head->next;
            head->next = cur->next;
            cur->next = pre->next;
            pre->next = cur;
        }
        return dumpHead->next;
    }

    //148. 排序链表
    /*
     bottom-to-up 的归并思路是这样的：先两个两个的 merge，完成一趟后，再 4 个4个的 merge，直到结束。举个简单的例子：[4,3,1,7,8,9,2,11,5,6].

     step=1: (3->4)->(1->7)->(8->9)->(2->11)->(5->6)
     step=2: (1->3->4->7)->(2->8->9->11)->(5->6)
     step=4: (1->2->3->4->7->8->9->11)->(5->6)
     step=8: (1->2->3->4->5->6->7->8->9->11)
     链表里操作最难掌握的应该就是各种断链啊，然后再挂接啊。在这里，我们主要用到链表操作的两个技术：

     merge(l1, l2)，双路归并，我相信这个操作大家已经非常熟练的，就不做介绍了。
     cut(l, n)，可能有些同学没有听说过，它其实就是一种 split 操作，即断链操作。不过我感觉使用 cut 更准确一些，它表示，将链表 l 切掉前 n 个节点，并返回后半部分的链表头。
     额外再补充一个 dummyHead 大法，已经讲过无数次了，仔细体会吧。
     */
    ListNode* sortList(ListNode* head) {
        ListNode *dumpHead = new ListNode(0);
        dumpHead->next = head;
        ListNode *p = head;
        int len = 0;
        while (p) {
            p = p->next;
            len++;
        }
        for (int i = 1; i < len; i = i * 2) {
            ListNode *cur = dumpHead->next;
            ListNode *tail = dumpHead;
            while (cur) {
                ListNode *left = cur;
                ListNode *right = cut(left, i);
                cur = cut(right, i);
                tail->next = megerNode(left, right);
                while (tail->next) {
                    tail = tail->next;
                }
            }
        }
        return dumpHead->next;
    }

    ListNode *cut(ListNode *node, int n) {
        ListNode *p = node;
        while (--n && p) {
            p = p->next;
        }
        if (!p) {
            return nullptr;
        }
        ListNode *next = p->next;
        p->next = nullptr;
        return next;
    }

    ListNode *megerNode(ListNode *l1, ListNode *l2) {
        ListNode *dumpHead = new ListNode(0);
        ListNode *p = dumpHead;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                p->next = l1;
                p = l1;
                l1 = l1->next;
            } else {
                p->next = l2;
                p = l2;
                l2 = l2->next;
            }
        }
        p->next = l1 ? l1 : l2;
        return dumpHead->next;
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

    int calculateMinimumHP2(vector<vector<int>>& dungeon) {
        int m = (int)dungeon.size();
        int n = (int)dungeon[0].size();
        vector<vector<int>> dp(m, vector<int>(n,0));
        for (int i = m-1; i >= 0; i--) {
            for (int j = n-1; j >= 0; j--) {
                if (i == m-1 && j == n-1) {
                    dp[i][j] = max(1, 1-dungeon[i][j]);
                } else if (i == m-1) {
                    dp[i][j] = max(1, dp[i][j+1]-dungeon[i][j]);
                } else if (j == n-1) {
                    dp[i][j] = max(1, dp[i+1][j]-dungeon[i][j]);
                } else {
                    dp[i][j] = max(1, min(dp[i+1][j],dp[i][j+1])-dungeon[i][j]);
                }
            }
        }
        return dp[0][0];
    }

    //189.旋转数组
    void rotate(vector<int>& nums, int k) {
        int n = (int)nums.size();
        k = k % n;
        if (k == 0) {
            return;
        }
        reverseArr(nums, 0, n-1);
        reverseArr(nums, 0, k-1);
        reverseArr(nums, k, n-1);
    }

    void reverseArr(vector<int>& nums, int l, int r) {
        while (l < r) {
            int temp = nums[l];
            nums[l++] = nums[r];
            nums[r--] = temp;
        }
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

    //209.长度最小的子数组
    //连续子数组和>=S  长度最小
    int minSubArrayLen(int s, vector<int>& nums) {
        int sum = 0;
        int len = 0;
        int left = 0;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            while (sum >= s) {
                if (len == 0) {
                    len = i-left+1;
                } else {
                    len = min(len, i-left+1);
                }
                sum -= nums[left++];
            }
        }
        return len;
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

    //215.数组中的第k个最大的元素
    int findKthLargest(vector<int>& nums, int k) {
        kthSort(nums, k, 0, nums.size());
        return nums[nums.size()-k];
    }

    void kthSort(vector<int>& nums, int k, int l, int r) {
        if (l >= r) {
            return;
        }
        int mid = quickKthSort(nums, l, r);
        if (mid == k) {
            return;
        } else if (mid > k) {
            kthSort(nums, k, l, mid-1);
        } else {
            kthSort(nums, k, mid+1, r);
        }
    }

    int quickKthSort(vector<int>& nums, int l, int r) {
        int tmp = nums[l];
        while (l < r) {
            while (l < r && nums[r] >= tmp) {
                r--;
            }
            nums[l] = nums[r];
            while (l < r && nums[l] <= tmp) {
                l++;
            }
            nums[r] = nums[l];
        }
        nums[l] = tmp;
        return l;
    }

    //221. 最大正方形
    int maximalSquare(vector<vector<char>>& matrix) {
        int res = 0;
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return res;
        }
        vector<vector<int>> dp(matrix.size(),vector<int>(matrix[0].size(),0));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = 1 + min(min(dp[i-1][j-1], dp[i][j-1]), dp[i-1][j]);
                    }
                    res = max(res, dp[i][j]);
                }
            }
        }
        return res * res;
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

    //229. 求众数 II
    vector<int> majorityElement2(vector<int>& nums) {
        int a = INT_MAX, b = INT_MAX, counta = 0, countb = 0;
        for (auto i : nums) {
            if (i == a) {
                counta++;
            } else if (i == b) {
                countb++;
            } else if (counta == 0) {
                a = i;
                counta++;
            } else if (countb == 0) {
                countb++;
            } else {
                counta--;
                countb--;
            }
        }
        counta = 0, countb = 0;
        for (auto i : nums) {
            if (i == a) {
                counta++;
            } else if (i == b) {
                countb++;
            }
        }
        vector<int> res;
        if (counta > nums.size() / 3) {
            res.push_back(a);
        }
        if (countb > nums.size() / 3) {
            res.push_back(b);
        }
        return res;
    }

    //230. 二叉搜索树中第K小的元素
    int kthSmallest(TreeNode* root, int k) {
        int res;
        int cur = 0;
        kthSmallestMidOrder(root, k, cur, res);
        return res;
    }

    void kthSmallestMidOrder(TreeNode *root, int k, int& cur, int & res) {
        if (!root) {
            return;
        }
        kthSmallestMidOrder(root->left, k, cur, res);
        cur += 1;
        if (cur == k) {
            res = root->val;
        }
        kthSmallestMidOrder(root->right, k, cur, res);
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

    //238.除自身以外数组的乘积
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> A(nums.size(), 1);
        vector<int> B(nums.size(), 1);
        for (int i = 1; i < nums.size(); i++) {
            A[i] = A[i-1] * nums[i-1];
        }
        for (int i = nums.size() - 2; i >= 0; i--) {
            B[i] = B[i+1] * nums[i+1];
        }
        for (int i = 0; i < nums.size(); i++) {
            A[i] *= B[i];
        }
        return A;
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

    //309. 最佳买卖股票时机含冷冻期
    int maxProfit2(vector<int>& prices) {
        int n = (int)prices.size();
        if (n <= 1) {
            return 0;
        }
        //分别代表截止到第i天最后一个操作是怎么样的收益 第0天只能是buy 收益是-prices[0]
        //第i天buy收益 只能有2种情况来 上一次也是buy即buy[i-1]或者是冷冻期之后的买入  cold[i-1] - prices[i]
        //第i天sell收益 上一次是sell即sell[i-1]或者是之前是持有 今天卖了 buy[i-1]+prices[i]
        //第i天cold收益 只能是由上一次是sell来
        vector<int> buy(n,0);
        vector<int> sell(n,0);
        vector<int> cold(n,0);
        buy[0] = -prices[0];
        for (int i = 1; i < n; i++) {
            sell[i] = max(buy[i-1] + prices[i], sell[i-1]);
            buy[i] = max(buy[i-1], cold[i-1] - prices[i]);
            cold[i] = sell[i-1];
        }
        return sell[n-1];
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

    //322.零钱兑换
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount+1,INT_MAX);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.size(); j++) {
                if (i >= coins[j]) {
                    dp[i] = min(dp[i], dp[i-coins[j]] + 1);
                }
            }
        }
        return dp[amount] == INT_MAX ? -1 : dp[amount];
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

    //350. 两个数组的交集 II
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        vector<int> res;
        unordered_map<int, int> map;
        for (auto i : nums1) {
            map[i]++;
        }
        for (auto i : nums2) {
            if (map[i] > 0) {
                res.push_back(i);
                map[i]--;
            }
        }
        return res;
    }

    //378. 有序矩阵中第K小的元素
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size() - 1;
        int l = matrix[0][0];
        int r = matrix[n][n];
        while (l < r) {
            int mid = l + (r-l)/2;
            int count = countOfKthSmallest(matrix, mid, n);
            if (count < k) {
                l = mid+1;
            } else {
                r = mid;
            }
        }
        return r;
    }

    int countOfKthSmallest(vector<vector<int>>& matrix, int mid, int n) {
        int x = n;
        int y = 0;
        int count = 0;
        while (x >= 0 && y <= n) {
            if (matrix[x][y] <= mid) {
                count += x + 1;
                y++;
            } else {
                x--;
            }
        }
        return count;
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

    //392.判断子序列
    bool isSubsequence(string s, string t) {
        int m = s.size();
        int n = t.size();
        if (m > n) {
            return false;
        }
        int s1 = 0;
        int t1 = 0;
        while (s1 < m && t1 < n) {
            if (s[s1] == t[t1]) {
                s1++;
                t1++;
            } else {
                t1++;
            }
        }
        return s1 == m;
    }

    //445. 两数相加 II
    //第2题两数相加是从头到尾 这个是从尾到头 所以用两个栈解决
    ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2) {
        stack<int> s1;
        stack<int> s2;
        while (l1) {
            s1.push(l1->val);
            l1 = l1->next;
        }
        while (l2) {
            s2.push(l2->val);
            l2 = l2->next;
        }
        int flag = 0;
        ListNode *head = nullptr;
        //注意进位大于0的时候也是需要的 比如两个链表都只有一个节点5 最后结果为1->0
        while (!s1.empty() || !s2.empty() || flag > 0) {
            int sum = flag;
            if (!s1.empty()) {
                sum += s1.top();
                s1.pop();
            }
            if (!s2.empty()) {
                sum += s2.top();
                s2.pop();
            }
            flag = sum / 10;
            ListNode *node = new ListNode(sum % 10);
            node->next = head;
            head = node;
        }
        return head;
    }

    //456.132模式 i < j < k 的时候 ai < ak < aj
    bool find132pattern(vector<int>& nums) {
        if (nums.size() < 3) {
            return false;
        }
        stack<int> stack;
        int last = INT_MIN; //第二大的值
        for (int i = (int)nums.size() - 1; i >= 0; i--) {
            if (nums[i] < last) {
                return true;
            }
            while (!stack.empty() && nums[i] > stack.top()) {
                last = stack.top();
                stack.pop();
            }
            stack.push(nums[i]);
        }
        return false;
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

    //503. 下一个更大元素 II
    /*
     遍历两次数组，把索引依次入栈，当后面遇到的num比当前栈顶的索引位置元素大的时候 就给res的索引位置元素赋值
     */
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = (int)nums.size();
        vector<int> res(n,-1);
        stack<int> stack;
        for (int i = 0; i < 2*n; i++) {
            int num = nums[i % n];
            while (!stack.empty() && num > nums[stack.top()]) {
                res[stack.top()] = num;
                stack.pop();
            }
            if (i < n) {
                stack.push(i);
            }
        }
        return res;
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

    //523. 连续的的子数组和
    //leetcode测试用例傻逼至极！！！！！！！
    bool checkSubarraySum(vector<int>& nums, int k) {
        map<int, int> map;
        map[0] = -1;
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            if (k != 0) {
                sum %= k;
            }
            if (map.count(sum)) {
                if (i - map[sum] > 1) {
                    return true;
                }
            } else {
                map[sum] = i;
            }
        }
        return false;
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

    //560. 和为K的子数组
    int subarraySum(vector<int>& nums, int k) {
        map<int, int> map;
        map[0] = 1;
        int count = 0;
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            int c = sum - k;
            if (map.find(c) != map.end()) {
                count++;
            }
        }
        return count;
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

    //565. 数组嵌套
    int arrayNesting(vector<int>& nums) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] < 0) {
                continue;
            }
            int cur = i;
            int count = 0;
            while (nums[cur] >= 0) {
                count++;
                int t = nums[cur];
                nums[cur] = -1;
                cur = t;
            }
            res = max(res, count);
        }
        return res;
    }

    //572.另一个🌲的子树
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (!s || !t) {
            return false;
        }
        return dfsSubTree(s, t) || isSubtree(s->left, t) || isSubtree(s->right, t);
    }

    bool dfsSubTree(TreeNode* s, TreeNode* t) {
        if (!s && !t) {
            return true;
        }
        if (s->val != t->val) {
            return false;
        }
        return dfsSubTree(s->left, t->left) && dfsSubTree(s->right, t->right);
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

    //670.最大交换
    int maximumSwap(int num) {
        vector<int> nums;
        while (num > 0) {
            int t = num % 10;
            nums.push_back(t);
            num /= 10;
        }
        for (int i = nums.size() - 1; i; --i){
            int choice = 0;
            for (int j = 0; j < i; ++j){
                if(nums[j] > nums[i] && nums[j] > nums[choice]){
                    choice = j;
                }
            }
            if(nums[choice] > nums[i]){
                swap(nums[choice], nums[i]);
                break;
            }
        }
        int res = 0;
        for (int i = nums.size()-1;i>=0;i--) {
            res *= 10;
            res += nums[i];
        }
        return res;
    }

    //674.最长连续递增序列
    int findLengthOfLCIS(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        int res = 1;
        int count = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i-1]) {
                count++;
                res = max(res, count);
            } else {
                count = 1;
            }
        }
        return res;
    }

    //680. 验证回文字符串 Ⅱ
    bool validPalindrome(string s) {
        return validString(s, 0, s.size()-1, false);
    }

    bool validString(string s, int left, int right, bool deleted) {
        while (left < right) {
            if (s[left] != s[right]) {
                if (deleted) {
                    return false;
                }
                return validString(s, left+1, right, true) || validString(s, left, right-1, true);
            }
            left++;
            right--;
        }
        return true;
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

    //695. 岛屿的最大面积
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    count = max(count, dfsMaxAreaOfIsland(grid, i, j));
                }
            }
        }
        return count;
    }

    int dfsMaxAreaOfIsland(vector<vector<int>>& grid, int row, int col) {
        if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size() || grid[row][col] == 0) {
            return 0;
        }
        grid[row][col] = 0;
        return 1 + dfsMaxAreaOfIsland(grid, row, col + 1) + dfsMaxAreaOfIsland(grid, row, col - 1) + dfsMaxAreaOfIsland(grid, row + 1, col) + dfsMaxAreaOfIsland(grid, row - 1, col);
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

    //714. 买卖股票的最佳的时机
    //买入再卖出会有一次手续费fee
    //cash 表示第i天不持有股票的利润 hold表示第i天持有股票的利润
    int maxProfit(vector<int>& prices, int fee) {
        if (prices.size() == 0) {
            return 0;
        }
        int cash = 0;
        int hold = -prices[0];
        for (int i = 1; i < prices.size(); i++) {
            cash = max(cash, hold + prices[i] - fee);
            hold = max(hold, cash - prices[i]);
        }
        return cash;
    }

    //718. 最长重复子数组
    int findLength(vector<int>& A, vector<int>& B) {
        vector<vector<int>> dp(A.size()+1,vector<int>(B.size()+1,0));
        int res = 0;
        for (int i = 1; i <= A.size(); i++) {
            for (int j = 1; j <= B.size(); j++) {
                if (A[i-1] == B[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                res = max(dp[i][j], res);
            }
        }
        return res;
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

    //735. 行星碰撞
    /*
     给定一个整数数组 asteroids，表示在同一行的行星。

     对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。

     找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。
     */
    //res保存保留的 如果最后一个是>0并且遇到小于0的 需要判断 然后是否进行pop 是否将小于0的push到res
    vector<int> asteroidCollision(vector<int>& asteroids) {
        vector<int> res;
        for (int i = 0; i < asteroids.size(); i++) {
            if (asteroids[i] > 0) {
                res.push_back(asteroids[i]);
            } else {
                bool flag = true;
                while (!res.empty() && res.back() > 0) {
                    if (abs(asteroids[i]) > res.back()) {
                        res.pop_back();
                        flag = true;
                    } else if (abs(asteroids[i]) == res.back()) {
                        res.pop_back();
                        flag = false;
                        break;
                    } else {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    res.push_back(asteroids[i]);
                }
            }
        }
        return res;
    }

    //739.每日温度
    vector<int> dailyTemperatures(vector<int>& T) {
        stack<int> stack;
        vector<int> res(T.size(), 0);
        for (int i = 0; i < T.size(); i++) {
            while (!stack.empty() && T[stack.top()] < T[i]) {
                res[stack.top()] = i - stack.top();
                stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    //740.删除与获得点数
    //变种型打家劫舍 把数组构建成一个包含数字个数的新数组
    int deleteAndEarn(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        if (nums.size() == 1) {
            return nums[0];
        }
        int maxNum = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            maxNum = max(maxNum, nums[i]);
        }
        vector<int> all(maxNum+1, 0);
        for (int i = 0; i < nums.size(); i++) {
            all[nums[i]]++;
        }
        vector<int> dp(maxNum+1,0);
        dp[0] = 0;
        dp[1] = all[1];
        for (int i = 2; i <= maxNum; i++) {
            dp[i] = max(dp[i-1],dp[i-2]+i*all[i]);
        }
        return dp[maxNum];
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

    //837.新21点
    double new21Game(int N, int K, int W) {
        //获胜条件：得分S<=N
        //K为0，不能抽，S必定<=N
        if (K == 0) return 1.0;
        //S为K-1时，抽到W，此为游戏结束时最大得分，如果都<=N，则也是必赢
        if (K+W-1 <= N) return 1.0;

        //由第二个条件取反，此时 N<K+W-1，就不用判断N+1跟K+W大小了
        vector<double> dp(K + W);
        //[K,N]为1，[N+1,K+W-1]为0
        for (int i = K; i <= N; i++) {
            dp[i] = 1.0;
        }
        //[K,N]长度N-K+1
        dp[K - 1] = 1.0 * (N - K + 1) / W;
        for (int i = K - 2; i >= 0; i--) {
            //dp[x]-dp[x+1]推导得来
            dp[i] = dp[i + 1] - (dp[i + W + 1] - dp[i + 1]) / W;
        }
        return dp[0];
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

    //887.鸡蛋掉落
    int superEggDrop(int K, int N) {
        if (N == 1) {
            return 1;
        }
        vector<vector<int>> f(N + 1, vector<int>(K + 1));
        for (int i = 1; i <= K; ++i) {
            f[1][i] = 1;
        }
        int ans = -1;
        for (int i = 2; i <= N; ++i) {
            for (int j = 1; j <= K; ++j) {
                f[i][j] = 1 + f[i - 1][j - 1] + f[i - 1][j];
            }
            if (f[i][K] >= N) {
                ans = i;
                break;
            }
        }
        return ans;
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

    //974.和可被K整除的子数组
    int subarraysDivByK(vector<int>& A, int K) {
        unordered_map<int, int> map{{0,1}};
        int sum = 0;
        int res = 0;
        for (auto num : A) {
            sum += num;
            int mod = (sum % K + K) % K;
            if (map.count(mod)) {
                res += map[mod];
            }
            ++map[mod];
        }
        return res;
    }

    //978. 最长湍流子数组
    int maxTurbulenceSize(vector<int>& A) {
        int up = 1;
        int down = 1;
        int res = 1;
        for (int i = 0; i < A.size()-1; i++) {
            if (A[i] > A[i+1]) {
                up = down + 1;
                down = 1;
            } else if (A[i] < A[i+1]) {
                down = up + 1;
                up = 1;
            } else {
                down = 1;
                up = 1;
            }
            res = max(max(up, down), res);
        }
        return res;
    }

    //979.在二叉树中分配硬币
    int distributeCoins(TreeNode* root) {
        int val = 0;
        coins(root, val);
        return val;
    }

    int coins(TreeNode *node, int &val) {
        if (node == nullptr) {
            return 0;
        }
        int left = coins(node->left, val);
        int right = coins(node->right, val);
        val += abs(left) + abs(right);
        return left + right + node->val - 1;
    }

    //1014 最佳观光组合
    /*
     给定正整数数组 A，A[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的距离为 j - i。
    一对景点（i < j）组成的观光组合的得分为（A[i] + A[j] + i - j）：景点的评分之和减去它们两者之间的距离。
    返回一对观光景点能取得的最高分
     */
    int maxScoreSightseeingPair(vector<int>& A) {
        int res = 0;
        int left = A[0];
        for (int i = 1; i < A.size(); i++) {
            res = max(res, A[i]-i+left);
            left = max(left,A[i]+i);
        }
        return res;
    }

    //1019. 链表中的下一个更大节点
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int> nums;
        while (head) {
            nums.push_back(head->val);
            head = head->next;
        }
        int n = (int)nums.size();
        vector<int> res(n,0);
        stack<int> stack;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            while (!stack.empty() && num > nums[stack.top()]) {
                res[stack.top()] = num;
                stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    //1028. 从先序遍历还原二叉树
    /*
     我们从二叉树的根节点 root 开始进行深度优先搜索。
    在遍历中的每个节点处，我们输出 D 条短划线（其中 D 是该节点的深度），然后输出该节点的值。（如果节点的深度为 D，则其直接子节点的深度为 D + 1。根节点的深度为 0）。
    如果节点只有一个子节点，那么保证该子节点为左子节点。
    给出遍历输出 S，还原树并返回其根节点 root。
     输入："1-2--3--4-5--6--7"
     输出：[1,2,5,3,4,6,7]
     */
    TreeNode* recoverFromPreorder(string S) {
        //d表示深度的queue v表示值
        queue<int> d;
        queue<int> v;
        int left = 0;
        for (int i = 0; i < S.size(); i++) {
            if (S[i] != '-') {
                d.push(i-left);
                left = i;
                while (i < S.size() && S[i] != '-') {
                    i++;
                }
                string s = S.substr(left,i-left);
                int val = atoi(s.c_str());
                v.push(val);
                left = i;
            }
        }
        TreeNode *node = nodeWithDepth(d, v, 0);
        return node;
    }

    TreeNode* nodeWithDepth(queue<int>& d, queue<int>& v, int depth) {
        if (d.front() != depth) {
            return nullptr;
        }
        TreeNode *node = new TreeNode(v.front());
        d.pop();
        v.pop();
        node->left = nodeWithDepth(d, v, depth+1);
        node->right = nodeWithDepth(d, v, depth+1);
        return node;
    }

    //1031.两个非重叠子数组的最大和
    // 左右两个子数组分别个数为L和M 使得总和最大
    int maxSumTwoNoOverlap(vector<int>& A, int L, int M) {
        int res = 0;
        int maxL = 0;
        int maxM = 0;
        vector<int> preSum(A.size()+1,0);
        for (int i = 0; i < A.size(); i++) {
            preSum[i+1] = A[i] + preSum[i];
        }
        //当L在左边的时候 从下标为L的位置开始遍历
        for (int i = L; i <= A.size()-M; i++) {
            maxL = max(maxL, preSum[i] - preSum[i-L]);
            res = max(res, maxL + (preSum[i+M]-preSum[i]));
        }
        //当M在左边的时候 从下标为M的位置开始遍历
        for (int i = M; i <= A.size()-L; i++) {
            maxM = max(maxM, preSum[i] - preSum[i-M]);
            res = max(res, maxM + (preSum[i+L]-preSum[i]));
        }
        return res;
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

    //1143.最长公共子序列
    //子序列是不连续的，子数组是连续的，注意区分
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i-1] == text2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
        return dp[m][n];
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

    //1371. 每个元音包含偶数次的最长子字符串
    int findTheLongestSubstring(string s) {
        // aeiou每个元音用一个bit一共5个bit，32种奇偶次数组合状态，比如10101可以表示aiu出现奇数次数
        // oe则出现偶数次数，每当遍历一个字符，就可以改变当前的aeiou出现的奇偶次数，也即是改变状态
        // 显然，如果两次出现了同样的状态，假设第一次出现在i处
        // 第二次出现在j处，那么i+1-j之间的字符串肯定是满足aeiou出现均为偶数次数的
        // 因为只有经历了偶数个aeiou，才能回到之前的状态，为了使得合理的字符串最长
        // 那么第一次出现此状态时，就需要记录到下标，然后下次遇到相同状态，计算最大长度
        unordered_map<int, int> map;
        int state = 0x0;
        int res = 0;
        // 初始化，刚开始时，state的状态已经是0x00000，已经出现，因此必须记录
        map[state] = 0;
        for (int i = 0; i < s.size(); i++) {
            // a e i o u 分别在第12345个bit，来表示出现次数的奇偶性
            if (s[i] == 'a') {
                state ^= 1<<0;
            } else if (s[i] == 'e') {
                state ^= 1<<1;
            } else if (s[i] == 'i') {
                state ^= 1<<2;
            } else if (s[i] == 'o') {
                state ^= 1<<3;
            } else if (s[i] == 'u') {
                state ^= 1<<4;
            }
            if (map.find(state) != map.end()) {
                res = max(res, i - map[state] + 1);
            } else {
                map[state] = i + 1;
            }
        }
        return res;
    }

    //1402.做菜顺序
    //排序之后从后往前加 每多加一次 之前的和再加一次 就表示几天了。
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(), satisfaction.end());
        int size = satisfaction.size();
        if (satisfaction[size - 1] < 0) {
            return 0;
        }
        int res = 0;
        int sum = 0;
        for (int i = size-1; i >= 0; i--) {
            sum += satisfaction[i];
            if (sum < 0) {
                break;
            }
            int t = sum + res;
            res = max(res, t);
        }
        return res;
    }

    //1414.和为k的最少斐波那契数目
    int findMinFibonacciNumbers(int k) {
        if (k <= 3) {
            return 1;
        }
        vector<int> nums;
        nums.push_back(1);
        nums.push_back(1);
        int sum = 0;
        int i = 2;
        while (sum < k) {
            sum = nums[i-1] + nums[i-2];
            nums.push_back(sum);
            i++;
        }
        int count = 0;
        for (int i = nums.size()-1; i >= 0; i--) {
            if (k >= nums[i]) {
                k -= nums[i];
                count++;
            }
            if (k == 0) {
                break;
            }
        }
        return count;
    }

    //1431. 拥有最多糖果的孩子
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        vector<bool> res(candies.size(), false);
        int max = 0;
        for (int i = 0; i < candies.size(); i++) {
            if (candies[i] >= max) {
                max = candies[i];
            }
        }
        for (int i = 0; i < candies.size(); i++) {
            if (candies[i] + extraCandies >= max) {
                res[i] = true;
            }
        }
        return res;
    }

    //1457. 二叉树中的伪回文路径
    int pseudoPalindromicPaths (TreeNode* root) {
        vector<int> route;
        int res = 0;
        searchRoute(root,route,res);
        return res;
    }

    void searchRoute(TreeNode *root, vector<int> &route, int & res) {
        if (root->left == nullptr && root->right == nullptr) {
            route.push_back(root->val);
            if (isValid(route)) {
                res += 1;
            }
            route.pop_back();
            return;
        }
        route.push_back(root->val);
        if (root->left) {
            searchRoute(root->left,route,res);
        }
        if (root->right) {
            searchRoute(root->right,route,res);
        }
        route.pop_back();
    }

    bool isValid(vector<int> route) {
        set<int> set;
        for (auto i : route) {
            if (set.find(i) != set.end()) {
                set.erase(i);
            } else {
                set.insert(i);
            }
        }
        return set.size() <= 1;
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
    //面试题01.01 判断字符是否唯一
    bool isUnique(string astr) {
        set<char> set;
        for (auto c : astr) {
            if (set.find(c) != set.end()) {
                return false;
            } else {
                set.insert(c);
            }
        }
        return true;
    }

    //面试题01.02 判断是否互为字符重排
    bool CheckPermutation(string s1, string s2) {
        if (s1.size() != s2.size()) {
            return false;
        }
        unordered_map<char, int> map;
        for (auto c : s1) {
            if (map.find(c) != map.end()) {
                map[c]++;
            } else {
                map[c] = 1;
            }
        }
        for (auto c : s2) {
            if (map.find(c) != map.end()) {
                map[c]--;
                if (map[c] == 0) {
                    map.erase(c);
                }
            } else {
                map[c] = 1;
            }
        }
        return map.size() == 0;
    }

    //面试题01.04 回文排列
    //判断字符是某个回文串的排列
    bool canPermutePalindrome(string s) {
        set<char> set;
        for (auto c : s) {
            if (set.find(c) != set.end()) {
                set.erase(c);
            } else {
                set.insert(c);
            }
        }
        return set.size() <= 1;
    }

    //面试题01.05 一次编辑
    bool oneEditAway(string first, string second) {
        int m = first.size();
        int n = second.size();
        if (abs(m-n) > 1) {
            return false;
        }
        int i = 0;
        int j = 0;
        bool edit = false;
        while (i < m && j < n) {
            if (first[i] == second[j]) {
                i++;
                j++;
            } else {
                if (edit) {
                    return false;
                }
                edit = true;
                if (m > n) {
                    i++;
                } else if (m < n) {
                    j++;
                } else {
                    i++;
                    j++;
                }
            }
        }
        if (edit && (i != m || j != n)) {
            return false;
        }
        return true;
    };

    //面试题01.06 字符串压缩
    string compressString(string S) {
        if (S.size() < 2) {
            return S;
        }
        string res;
        int i = 0;
        char c = S[0];
        int count = 0;
        while (i < S.size()) {
            if (c != S[i]) {
                res.push_back(c);
                res += to_string(count);
                c = S[i];
                count = 1;
            } else {
                count++;
            }
            if (i == S.size()-1) {
                res.push_back(c);
                res += to_string(count);
            }
            i++;
        }
        if (res.size() > S.size()) {
            return S;
        } else {
            return res;
        }
    }

    //面试题01.08 零矩阵
    //把0所在的行列都置为0
    void setZeroes(vector<vector<int>>& matrix) {
        vector<vector<int>> tmp = matrix;
        for (int i = 0; i < tmp.size(); i++) {
            for (int j = 0; j < tmp[0].size(); j++) {
                if (tmp[i][j] == 0) {
                    transformZero(matrix, i, j);
                }
            }
        }
    }

    void transformZero(vector<vector<int>>& matrix, int row, int col) {
        for (int i = 0; i < matrix.size(); i++) {
            matrix[i][col] = 0;
        }
        for (int i = 0; i < matrix[0].size(); i++) {
            matrix[row][i] = 0;
        }
    }

    //面试题01.09 字符串轮转
    //abcd bcda
    bool isFlipedString(string s1, string s2) {
        if (s1.size() != s2.size()) {
            return false;
        }
        int len = s1.size();
        s2 += s2;
        for (int i = 0; i < len; i++) {
            if (s1[0] == s2[i]) {
                string tmp = s2.substr(i,len);
                if (tmp == s1) {
                    return true;
                }
            }
        }
        return false;
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

    //面试题02.04 分割链表
    //小于x的都位于大于x的左边
    ListNode* partition2(ListNode* head, int x) {
        ListNode * less = new ListNode(0);
        ListNode * more = new ListNode(0);
        ListNode *curLess = less;
        ListNode *curMore = more;
        while (head) {
            if (head->val < x) {
                curLess->next = head;
                curLess = curLess->next;
            } else {
                curMore->next = head;
                curMore = curMore->next;
            }
            head = head->next;
        }
        curMore->next = nullptr;
        curLess->next = more->next;
        return less->next;
    }

    //面试题 02.06. 回文链表
    bool isPalindrome2(ListNode* head) {
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        slow = reverseListNode(slow);
        while (head && slow) {
            if (head->val != slow->val) {
                return false;
            }
            head = head->next;
            slow = slow->next;
        }
        return true;
    }

    ListNode *reverseListNode(ListNode *node) {
        ListNode *newH = nullptr;
        for (ListNode *p = node; p; ) {
            ListNode *tmp = p->next;
            p->next = newH;
            newH = p;
            p = tmp;
        }
        return newH;
    }

    //面试题02.07 链表相交
    ListNode *getIntersectionNode2(ListNode *headA, ListNode *headB) {
        ListNode *A = headA;
        ListNode *B = headB;
        while (A != B) {
            A = A == nullptr ? headB : A->next;
            B = B == nullptr ? headA : B->next;
        }
        return A;
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

    //面试题04.01 节点间通路
    bool findWhetherExistsPath(int n, vector<vector<int>>& graph, int start, int target) {
        vector<vector<int>> v(n);
        vector<bool> visited(n,false);
        for (auto i : graph) {
            v[i[0]].push_back(i[1]);
        }
        return dfsFindWhetherExistsPath(v, visited, start, target);
    }

    bool dfsFindWhetherExistsPath(vector<vector<int>>& v, vector<bool> & visited, int start, int target) {
        if (start == target) {
            return true;
        }
        visited[start] = true;
        for (auto i : v[start]) {
            if (!visited[i]) {
                if (dfsFindWhetherExistsPath(v, visited, i, target)) {
                    return true;
                }
            }
        }
        return false;
    }

    //面试题04.02 最小高度🌲
    TreeNode* sortedArrayToBST2(vector<int>& nums) {
        int n = nums.size();
        return sortedArrayToBST2(nums, 0, n-1);
    }

    TreeNode *sortedArrayToBST2(vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        int mid = left + (right-left+1)/2;
        TreeNode *node = new TreeNode(nums[mid]);
        node->left = sortedArrayToBST2(nums, left, mid-1);
        node->right = sortedArrayToBST2(nums, mid+1, right);
        return node;
    }

    //面试题 04.03. 特定深度节点链表
    vector<ListNode*> listOfDepth(TreeNode* tree) {
        vector<ListNode *> res;
        queue<TreeNode *> q;
        if (!tree) {
            return res;
        }
        q.push(tree);
        while (!q.empty()) {
            int s = q.size();
            vector<int> tmp;
            while (s > 0) {
                TreeNode *node = q.front();
                tmp.push_back(node->val);
                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
                q.pop();
                s--;
            }
            ListNode *list = vectorToList(tmp);
            res.push_back(list);
        }
        return res;
    }

    ListNode *vectorToList(vector<int>& nums) {
        ListNode *node = new ListNode(-1);
        ListNode *h = node;
        for (auto i : nums) {
            ListNode *t = new ListNode(i);
            h->next = t;
            h = h->next;
        }
        return node->next;
    }

    //面试题 04.05. 合法二叉搜索树
    //中序遍历 升序
    bool isValidBST(TreeNode* root) {
        return midTraverse(root, LONG_MIN, LONG_MAX);
    }

    bool midTraverse(TreeNode *root, long min, long max) {
        if (!root) {
            return true;
        }
        if (root->val > min && root->val < max) {
            return midTraverse(root->left, min, root->val) && midTraverse(root->right, root->val, max);
        }
        return false;
    }

    //面试题 04.06. 后继者
    //二叉搜索树中某个节点的后续节点
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        vector<TreeNode *> res;
        midNode(root, res);
        for (int i = 0; i < res.size(); i++) {
            if (res[i]->val == p->val && i != res.size()-1) {
                return res[i+1];
            }
        }
        return nullptr;
    }

    void midNode(TreeNode *root, vector<TreeNode *>& res) {
        midNode(root->left, res);
        res.push_back(root);
        midNode(root->right, res);
    }

    //面试题 04.08. 首个共同祖先
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) {
            return root;
        }
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (left && right) {
            return root;
        }
        if (left && !right) {
            return left;
        }
        return right;
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

    //面试题08.07无重复字符串的排列组合
    vector<string> permutation33(string S) {
        vector<string> res;
        vector<bool> visited(S.size(),false);
        string p = "";
        dfsPermutation33(S, p, res, visited);
        return res;
    }

    void dfsPermutation33(string &s, string &p, vector<string> &res, vector<bool> &visited) {
        if (p.size() == s.size()) {
            res.push_back(p);
            return;
        }
        for (int i = 0; i < s.size(); i++) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            p.push_back(s[i]);
            dfsPermutation33(s, p, res, visited);
            p.pop_back();
            visited[i] = false;
        }
    }


    //面试题08.10 颜色填充
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        if (image[sr][sc] == newColor) {
            return image;
        }
        dfsFloodFill(image, sr, sc, newColor, image[sr][sc]);
        return image;
    }

    void dfsFloodFill(vector<vector<int>>& image, int sr, int sc, int newColor, int oldColor) {
        if (sr < 0 || sr >= image.size() || sc < 0 || sc >= image[0].size()) {
            return;
        }
        if (image[sr][sc] == oldColor) {
            image[sr][sc] = newColor;
            dfsFloodFill(image, sr+1, sc, newColor, oldColor);
            dfsFloodFill(image, sr-1, sc, newColor, oldColor);
            dfsFloodFill(image, sr, sc+1, newColor, oldColor);
            dfsFloodFill(image, sr, sc-1, newColor, oldColor);
        }
    }

    //面试题08.11 分硬币
    int waysToChange(int n) {
        vector<int> dp(n+1,0);
        vector<int> coins{1,5,10,25};
        dp[0] = 1;
        for (int i = 0; i < coins.size(); i++) {
            for (int j = coins[i]; j <= n; j++) {
                dp[j] = (dp[j] + dp[j-coins[i]]) % 1000000007;
            }
        }
        return dp[n];
    }

    //面试题08.12 N皇后
    vector<vector<string>> solveNQueens22(int n) {
        vector<vector<string>> res;
        vector<string> queen(n,string(n,'.'));
        dfsQueen(res, queen, 0);
        return res;
    }

    void dfsQueen (vector<vector<string>> &res, vector<string> &queen, int row) {
        if (row == queen.size()) {
            res.push_back(queen);
            return;
        }
        for (int col = 0; col < queen.size(); col++) {
            if (canQueen(queen, row, col)) {
                queen[row][col] = 'Q';
                dfsQueen(res, queen, row+1);
                queen[row][col] = '.';
            }
        }
    }

    bool canQueen(vector<string> queen, int row, int col) {
        //判断列
        for (int i = 0; i < row; i++) {
            if (queen[i][col] == 'Q') {
                return false;
            }
        }
        //判断左上
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--,j--) {
            if (queen[i][j] == 'Q') {
                return false;
            }
        }
        //判断右上
        for (int i = row - 1, j = col + 1; i >= 0 && j < queen.size(); i--,j++) {
            if (queen[i][j] == 'Q') {
                return false;
            }
        }
        return true;
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

    //面试题10.01 合并排序的数组
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        while (m > 0 || n > 0) {
            if (m == 0) {
                A[m+n-1] = B[n-1];
                n--;
            } else if (n == 0) {
                A[m+n-1] = A[m-1];
                m--;
            } else {
                if (A[m-1] > B[n-1]) {
                    A[m+n-1] = A[m-1];
                    m--;
                } else {
                    A[m+n-1] = B[n-1];
                    n--;
                }
            }
        }
    }

    //面试题 10.09. 排序矩阵查找
    bool searchMatrix2(vector<vector<int>>& matrix, int target) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return false;
        }
        int m = (int)matrix.size();
        int n = (int)matrix[0].size();
        if (matrix[0][0] > target || matrix[m-1][n-1] < target) {
            return false;
        }
        int row = m-1;
        int col = 0;
        while (row >= 0 && col <= n-1) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                row--;
            } else {
                col++;
            }
        }
        return false;
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

    //面试题16.11 跳水板
    vector<int> divingBoard(int shorter, int longer, int k){
        vector<int> res;
        if (k == 0) {
            return res;
        }
        if (shorter == longer) {
            return vector<int>{shorter * k};
        }
        for (int i = 0; i <= k; i++) {
            int length = longer * i + (k-i) * shorter;
            res.push_back(length);
        }
        return res;
    }

    vector<int> divingBoard2(int shorter, int longer, int k) {
        if (k == 0) {
            return vector<int>{0};
        }
        if (shorter == longer) {
            return vector<int>{longer * k};
        }
        vector<int> res;
        int t = k;
        while (t >= 0) {
            int n = shorter * t + (k-t)*longer;
            res.push_back(n);
        }
        return res;
    }

    //面试题 17.13. 恢复空格
    //dp 遍历到i位置的时候依次在dict里面找是否存在相应的string， 如果找到了 就dp[i+1] = min(dp[i+1],dp[i+1-len])
    int respace(vector<string>& dictionary, string sentence) {
        int n = (int)sentence.size();
        vector<int> dp(n+1);
        dp[0] = 0;
        for (int i = 0; i < n; i++) {
            dp[i+1] = dp[i]+1;
            for (auto str: dictionary) {
                int len = (int)str.size();
                if (len <= i+1) {
                    if (sentence.substr(i+1-len,len) == str) {
                        dp[i+1] = min(dp[i+1], dp[i+1-len]);
                    }
                }
            }
        }
        return dp[n];
    }

    //面试题 17.18. 最短超串
    vector<int> shortestSeq(vector<int>& big, vector<int>& small) {
        unordered_map<int, int> needs;
        unordered_map<int, int> windows;
        int min = INT_MAX;
        int start = 0;
        vector<int> res;
        for (auto i : small) {
            needs[i]++;
        }
        int left = 0;
        int right = 0;
        int valid = 0;
        while (right < big.size()) {
            if (needs.count(big[right])) {
                windows[big[right]]++;
                if (windows[big[right]] == needs[big[right]]) {
                    valid++;
                }
            }
            right++;
            while (valid == small.size()) {
                if (right - left < min) {
                    min = right - left;
                    start = left;
                }
                int c = big[left];
                left++;
                if (needs.count(c)) {
                    if (windows[c] == needs[c]) {
                        valid--;
                    }
                    windows[c]--;
                }
            }
        }
        if (min == INT_MAX) {
            return res;
        }
        res.push_back(start);
        res.push_back(start+min-1);
        return res;
    }

    //面试题17.19 消失的2个数字
    vector<int> missingTwo(vector<int>& nums) {
        int n = nums.size() + 2;
        int a = 0;
        int b = 0;
        int c = 0;
        for (int i = 1; i <= n; i++) {
            c ^= i;
        }
        for (auto i : nums) {
            c ^= i;
        }
        int h = 1;
        while ((c & h) == 0) {
            h <<= 1;
        }
        for (int i = 1; i <= n; i++) {
            if ((i & h) == 0) {
                a ^= i;
            } else {
                b ^= i;
            }
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

    //面试题40. 最小的k个数
    //快排找到指定下标k-1的数
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        if (k == 0 || arr.size() == 0) {
            return res;
        }
        sortQuickNums(arr, k, 0, arr.size()-1);
        for (int i = 0; i < k; i++) {
            res.push_back(arr[i]);
        }
        return res;
    }

    void sortQuickNums(vector<int>& arr,int k, int left, int right) {
        int index = quickSortP(arr, left, right);
        if (index == k) {
            return;
        } else if (index < k) {
            sortQuickNums(arr, k, index+1, right);
        } else {
            sortQuickNums(arr, k, left, index-1);
        }
    }

    int quickSortP(vector<int>& arr, int left, int right) {
        int tmp = arr[left];
        while (left < right) {
            while (left < right && arr[right] >= tmp) {
                right--;
            }
            arr[left] = arr[right];
            while (left < right && arr[left] <= tmp) {
                left++;
            }
            arr[right] = arr[left];
        }
        arr[left] = tmp;
        return left;
    }

    //面试题44. 数字序列中某一位的数字
    int findNthDigit(int n) {
        // 计算该数字由几位数字组成，由1位：digits = 1；2位：digits = 2...
        long base = 9,digits = 1;
        while (n - base * digits > 0){
            n -= base * digits;
            base *= 10;
            digits ++;
        }

        // 计算真实代表的数字是多少
        int idx = n % digits;  // 注意由于上面的计算，n现在表示digits位数的第n个数字
        if (idx == 0)idx = digits;
        long number = 1;
        for (int i = 1;i < digits;i++)
            number *= 10;
        number += (idx == digits)? n/digits - 1:n/digits;

        // 从真实的数字中找到我们想要的那个数字
        for (int i=idx;i<digits;i++) number /= 10;
        return number % 10;
    }

    //面试题46. 把数字翻译成字符串
    int translateNum(int num) {
        if (num == 0) {
            return 1;
        }
        vector<int> nums;
        while (num > 0) {
            nums.push_back(num%10);
            num /= 10;
        }
        vector<int> dp(nums.size()+1,0);
        dp[nums.size()] = 1;
        dp[nums.size() - 1] = 1;
        for (int i = nums.size()-2; i >= 0; i--) {
            int tmp = nums[i] + nums[i+1] * 10;
            if (tmp <= 25 && tmp >= 10) {
                dp[i] = dp[i+1] + dp[i+2];
            } else {
                dp[i] = dp[i+1];
            }
        }
        return dp[0];
    }

    //面试题48. 最长不含重复字符的子字符串
    int lengthOfLongestSubstring2(string s) {
        if (s.size() <= 2) {
            return s.size();
        }
        map<char, int> map;
        int res = 0;
        int left = 0;
        for (int i = 0; i < s.size(); i++) {
            int index = -1;
            if (map.find(s[i]) != map.end()) {
                index = map[s[i]];
            }
            if (index >= left) {
                left = index + 1;
            } else {
                res = max(res, i - left + 1);
            }
            map[s[i]] = i;
        }
        return res;
    }

    //面试题50. 第一个只出现一次的字符
    char firstUniqChar(string s) {
        map<char, int> map;
        for (int i = 0; i < s.size(); i++) {
            map[s[i]]++;
        }
        for (int i = 0; i < s.size(); i++) {
            if (map[s[i]] == 1) {
                return s[i];
            }
        }
        return ' ';
    }

    //面试题51. 数组中的逆序对
    int reversePairs(vector<int>& nums) {
        int res = 0;
        vector<int> tmp(nums.size(),0);
        megerSort2(nums, 0, nums.size()-1, tmp, res);
        return res;
    }

    void megerSort2(vector<int>& nums, int left, int right, vector<int>& tmp, int& res) {
        if (left >= right) {
            return;
        }
        int mid = (right-left)/2 + left;
        megerSort2(nums, left, mid, tmp, res);
        megerSort2(nums, mid+1, right, tmp, res);
        meger2(nums, left, mid, right, tmp, res);
    }

    void meger2(vector<int>& nums, int left, int mid, int right, vector<int>& tmp, int& res) {
        int i = left;
        int j = mid+1;
        int k = 0;
        while (i <= mid && j <= right) {
            if (nums[i] > nums[j]) {
                tmp[k++] = nums[j++];
                res += mid-i+1;
            } else {
                tmp[k++] = nums[i++];
            }
        }
        while (i <= mid) {
            tmp[k++] = nums[i++];
        }
        while (j <= right) {
            tmp[k++] = nums[j++];
        }
        for (int index = 0; left <= right; left++,index++) {
            nums[left] = tmp[index];
        }
    }

    //面试题53 - I. 在排序数组中查找数字 I
    int searchTarget(vector<int>& nums, int target) {
        if (nums.size() == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.size() - 1;
        int count = 0;
        while (left < right) {
            int mid = left + (right - left)/2;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        while (left < nums.size() && nums[left++] == target) {
            count++;
        }
        return count;
    }

    //面试题53 - II. 0～n-1中缺失的数字
    int missingNumber(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        while (left < right) {
            int mid = (right-left)/2+left;
            if (nums[mid] == mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (nums[left] == left && left == nums.size() - 1) {
            return left+1;
        } else {
            return left;
        }
    }

    //面试题54. 二叉搜索树的第k大节点
    int kthLargest(TreeNode* root, int k) {
        int res = 0;
        int count = 0;
        helperKthLargest(root, k, res, count);
        return res;
    }

    void helperKthLargest(TreeNode *root, int k, int& res, int& count) {
        if (root->right) {
            helperKthLargest(root->right, k, res, count);
        }
        if (++count == k) {
            res = root->val;
            return;
        }
        if (root->left) {
            helperKthLargest(root->left, k, res, count);
        }
    }

    //面试题55 - I. 二叉树的深度
    int maxDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        return max(maxDepth(root->left),maxDepth(root->right)) + 1;
    }

    //面试题55 - II. 平衡二叉树
    bool isBalanced(TreeNode* root) {
        if (!root) {
            return true;
        }
        if (abs(maxDepth(root->left) - maxDepth(root->right)) <= 1) {
            return isBalanced(root->left) && isBalanced(root->right);
        }
        return false;
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

    vector<int> singleNumbers2(vector<int>& nums) {
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
            if (i&h) {
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

    //面试题59 - I. 滑动窗口的最大值
    //双端队列保存最大值的下标
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> res;
        for (int i = 0; i < nums.size(); i++) {
            while (!dq.empty() && i - dq.front() + 1 > k) {
                dq.pop_front();
            }
            while (!dq.empty() && nums[dq.back()] < nums[i]) {
                dq.pop_back();
            }
            dq.push_back(i);
            if (i >= k-1) {
                res.push_back(nums[dq.front()]);
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

    //面试题66.构建乘积数组
    //从左遍历一次 再从右遍历一次
    vector<int> constructArr(vector<int>& a) {
        vector<int> res(a.size(),1);
        int tmp = 1;
        for (int i = 0; i < a.size(); i++) {
            res[i] = tmp;
            tmp *= a[i];
        }
        tmp = 1;
        for (int i = a.size()-1; i >= 0; i--) {
            res[i] *= tmp;
            tmp *= a[i];
        }
        return res;
    }

    //排列字符串包含去重 再写一遍
    vector<string> permutation22(string s) {
        vector<string> res;
        sort(s.begin(), s.end());
        vector<bool> visited(s.size(),false);
        dfspermutation22(s, visited, res, "");
        return res;
    }

    void dfspermutation22(string s, vector<bool> visited, vector<string>& res, string p) {
        if (p.size() == s.size()) {
            res.push_back(p);
            return;
        }
        for (int i = 0; i < s.size(); i++) {
            if (visited[i]) {
                continue;
            }
            if (i > 0 && s[i-1] == s[i] && !visited[i-1]) {
                continue;
            }
            visited[i] = true;
            p.push_back(s[i]);
            dfspermutation22(s, visited, res, p);
            visited[i] = false;
            p.pop_back();
        }
    }

    //复习快排 再写一遍
    void sortArray22(vector<int>& nums, int left, int right) {
        if (left >= right) {
            return;
        }
        int index = quickSort22(nums, left, right);
        sortArray22(nums, left, index-1);
        sortArray22(nums, index+1, right);
    }

    int quickSort22(vector<int>& nums, int left, int right) {
        int tmp = nums[left];
        while (left < right) {
            while (left < right && nums[right] >= tmp) {
                right--;
            }
            nums[left] = nums[right];
            while (left < right&& nums[left] <= tmp) {
                left++;
            }
            nums[right] = nums[left];
        }
        nums[left] = tmp;
        return left;
    }
    //复习插入排序 再写一遍
    void insetSort(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            int insert = nums[i];
            int j = i;
            while (j > 0 && nums[j-1] > insert) {
                nums[j] = nums[j-1];
                j--;
            }
            nums[j] = insert;
        }
    }

    //复习希尔排序
    void shellSort(vector<int>& nums, int count) {
        int interval = count / 2;
        while (interval > 0) {
            for (int i = interval; i < count; i++) {
                int insert = nums[i];
                int j = i-interval;
                while (j >= 0 && nums[j] > insert) {
                    nums[j+interval] = nums[j];
                    j -= interval;
                }
                nums[j+interval] = insert;
            }
            interval /= 2;
        }
    }

    //复习归并排序
    void megerSort(vector<int>& nums, int left, int right, vector<int>& tmp) {
        if (left < right) {
            int mid = left + (right - left)/2;
            megerSort(nums, left, mid, tmp);
            megerSort(nums, mid+1, right, tmp);
            meger(nums, left, mid, right, tmp);
        }
    }

    void meger(vector<int>& nums, int left, int mid, int right, vector<int>& tmp) {
        int i = left;
        int j = mid+1;
        int k = 0;
        while (i <= mid && j <= right) {
            if (nums[i] > nums[j]) {
                tmp[k++] = nums[j++];
            } else {
                tmp[k++] = nums[i++];
            }
        }
        while (i <= mid) {
            tmp[k++] = nums[i++];
        }
        while (j <= right) {
            tmp[k++] = nums[j++];
        }
        for (int index = 0; left <= right; left++,index++) {
            nums[left] = tmp[index];
        }
    }

    //复习堆排序
    void heapSort(vector<int>& nums, int len) {
        //先构建成最大堆
        for (int i = len/2; i >= 0; i--) {
            buildHeapArr(nums, i, len);
        }
        for (int i = len-1; i >= 0; i--) {
            //每次把当前最大堆中的堆顶元素和最后一个元素交换，再对剩下的i-1个元素进行构建最大堆  这样到最后出来的就是升序
            int tmp = nums[0];
            nums[0] = nums[i];
            nums[i] = tmp;
            buildHeapArr(nums, 0, i);
        }
    }

    void buildHeapArr(vector<int>& nums, int i, int len) {
        int left = 2*i+1;
        int right = 2*i+2;
        int max = i;
        if (left < len && nums[left] > nums[max]) {
            max = left;
        }
        if (right < len && nums[right] > nums[max]) {
            max = right;
        }
        if (i != max) {
            int tmp = nums[i];
            nums[i] = nums[max];
            nums[max] = tmp;
            buildHeapArr(nums, max, len);
        }
    }

    //二叉树遍历 迭代 前、中、后，用栈访问，看哪个节点出栈的时候保存值，后序需要判断root的之前访问节点如果是右节点或者右节点为空才会出栈
    vector<int> iterationPreOrder(TreeNode *root) {
        vector<int> res;
        stack<TreeNode *> stack;
        while (root || !stack.empty()) {
            while (root) {
                stack.push(root);
                res.push_back(root->val);
                root = root->left;
            }
            while (!root && !stack.empty()) {
                root = stack.top()->right;
                stack.pop();
            }
        }
        return res;
    }

    vector<int> iterationInOrder(TreeNode *root) {
        vector<int> res;
        stack<TreeNode *> stack;
        while (root || !stack.empty()) {
            while (root) {
                stack.push(root);
                root = root->left;
            }
            root = stack.top();
            res.push_back(root->val);
            stack.pop();
            root = root->right;
        }
        return res;
    }

    vector<int> iterationPostOrder(TreeNode *root) {
        vector<int> res;
        stack<TreeNode *> stack;
        TreeNode *pre = nullptr;
        while (root || !stack.empty()) {
            while (root) {
                stack.push(root);
                root = root->left;
            }
            root = stack.top();
            if (root->right == nullptr || root->right == pre) {
                pre = root;
                res.push_back(root->val);
                stack.pop();
                root = nullptr;
            } else {
                root = root->right;
            }
        }
        return res;
    }

    bool isMatch2(string s, string p) {
        int m = s.size();
        int n = p.size();
        if (m == 0 && n == 0) {
            return true;
        }
        vector<vector<bool>> dp(m,vector<bool>(n,false));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (i>=2 && p[i-1] == '*' && p[i-2]) {
                dp[0][i] = dp[0][i-2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i-1] == p[j-1] || p[j-1] == '.') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p[i-1] == '*') {
                    if (s[i-1] != p[j-2] && p[j-2] != '.') {
                        dp[i][j] = dp[i][j-2];
                    } else {
                        dp[i][j] = dp[i][j-2] || dp[i][j-1] || dp[i-1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }


    bool isMatch3(string s, string p) {
        int m = s.size();
        int n = p.size();
        if (m == 0 && n == 0) {
            return true;
        }
        vector<vector<bool>> dp(m+1,vector<bool>(n+1,false));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (i >= 2 && p[i-1] == '*' && p[i-2]) {
                dp[0][i] = dp[0][i-2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s[i-1] == p[j-1] || p[j-1] == '.') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p[j-1] == '*') {
                    if (p[j-2] != s[i-1] && p[j-2] != '.') {
                        dp[i][j] = dp[i][j-2];
                    } else {
                        dp[i][j] = dp[i][j-1] || dp[i-1][j] || dp[i][j-2];
                    }
                }
            }
        }
        return dp[m][n];
    }

    int longestConsecutive2(vector<int> & nums) {
        map<int, int> map;
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (map.find(nums[i]) == map.end()) {
                int left = nums[i]-1;
                int right = nums[i]+1;
                int len1 = 0;
                if (map.find(left) != map.end()) {
                    len1 = map[left];
                }
                int len2 = 0;
                if (map.find(right) != map.end()) {
                    len2 = map[right];
                }
                int len = len1 + len2 + 1;
                res = max(res,len);
                map[nums[i]] = len;
                map[nums[i]-len1] = len;
                map[nums[i]+len2] = len;
            }
        }
        return res;
    }

    //5.31周赛
    int maxProduct(vector<int>& nums) {
        if (nums.size() < 2) {
            return 0;
        }
        int maxNum = max(nums[0],nums[1]);
        int minNum = min(nums[0],nums[1]);
        for (int i = 2; i < nums.size(); i++) {
            if (nums[i] > minNum) {
                maxNum = max(maxNum, nums[i]);
                minNum = min(maxNum, nums[i]);
            }
        }
        return (maxNum-1) * (minNum-1);
    }

    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts) {
        sort(horizontalCuts.begin(), horizontalCuts.end());
        sort(verticalCuts.begin(), verticalCuts.end());
        int maxW = verticalCuts[0];
        int maxH = horizontalCuts[0];
        if (verticalCuts.size() == 1) {
            maxW = max(w-verticalCuts[0],verticalCuts[0]);
        }
        if (horizontalCuts.size() == 1) {
            maxH = max(h-horizontalCuts[0],horizontalCuts[0]);
        }
        for (int i = 1; i < horizontalCuts.size(); i++) {
            maxH = max(maxH, horizontalCuts[i] - horizontalCuts[i-1]);
            if (i == horizontalCuts.size() - 1) {
                maxH = max(h - horizontalCuts[i],maxH);
            }
        }
        for (int i = 1; i < verticalCuts.size(); i++) {
            maxW = max(maxW, verticalCuts[i] - verticalCuts[i-1]);
            if (i == verticalCuts.size() - 1) {
                maxW = max(w - verticalCuts[i],maxW);
            }
        }
        return (maxW %(10^9 + 7))* (maxH%(10^9 + 7));
    }

    int longestConsecutive3(vector<int>& nums) {
        map<int, int> map;
        int res = 0;
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
                res = max(res, len);
                map[nums[i]] = len;
                map[nums[i]-len1] = len;
                map[nums[i]+len2] = len;
            }
        }
        return res;
    }

    //6.7周赛
    vector<int> shuffle(vector<int>& nums, int n) {
        vector<int> res;
        for (int i = 0; i < n; i++) {
            res.push_back(nums[i]);
            res.push_back(nums[i+n]);
        }
        return res;
        int x = 0;
        int y = n;
        int tempx = nums[x];
        int tempy = nums[y];
        for (int i = 0; i < nums.size(); i++) {
            if (i % 2==0) {
                int t = nums[++x];
                nums[i] = tempx;
                tempx = t;
            } else {
                int t = nums[++y];
                nums[i] = tempy;
                tempy = t;
            }
        }
        return nums;
    }

    vector<int> getStrongest(vector<int>& arr, int k) {
        sort(arr.begin(), arr.end());
        int mid = arr[(arr.size() - 1)/2];
        vector<int> res;
        int left = 0;
        int right = arr.size()-1;
        while (k > 0) {
            if (arr[right] - mid >= mid - arr[left]) {
                res.push_back(arr[right]);
                right--;
            } else {
                res.push_back(arr[left]);
                left++;
            }
            k--;
        }
        return res;
    }

    //6.13双周赛
    //5420. 商品折扣后的最终价格
    vector<int> finalPrices(vector<int>& prices) {
        for (int i = 0; i < prices.size(); i++) {
            for (int j = i + 1; j < prices.size(); j++) {
                if (prices[j] <= prices[i]) {
                    prices[i] -= prices[j];
                    break;
                }
            }
        }
        return prices;
    }

    //5422. 子矩形查询
    /*
     请你实现一个类 SubrectangleQueries ，它的构造函数的参数是一个 rows x cols 的矩形（这里用整数矩阵表示），并支持以下两种操作：

     1. updateSubrectangle(int row1, int col1, int row2, int col2, int newValue)

     用 newValue 更新以 (row1,col1) 为左上角且以 (row2,col2) 为右下角的子矩形。
     2. getValue(int row, int col)

     返回矩形中坐标 (row,col) 的当前值。
     */
    class SubrectangleQueries {
        vector<vector<int>> matrix;
    public:
        SubrectangleQueries(vector<vector<int>>& rectangle) {
            matrix = rectangle;
        }

        void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
            for (int i = row1; i <= row2; i++) {
                for (int j = col1; j <= col2; j++) {
                    matrix[i][j] = newValue;
                }
            }
        }

        int getValue(int row, int col) {
            return matrix[row][col];
        }
    };

    //5423. 找两个和为目标值且不重叠的子数组
    /*
     给你一个整数数组 arr 和一个整数值 target 。

     请你在 arr 中找 两个互不重叠的子数组 且它们的和都等于 target 。可能会有多种方案，请你返回满足要求的两个子数组长度和的 最小值 。

     请返回满足要求的最小长度和，如果无法找到这样的两个子数组，请返回 -1 。
     */
    //not done
    int minSumOfLengths(vector<int>& arr, int target) {
        unordered_map<int, int> map;
        map[0] = 0;
        int sum = 0;
        for (int i = 0; i < arr.size(); i++) {
            sum += arr[i];
            if (map.count(sum-target)) {

            }
        }
        return 1;
    }

    //6.14 周赛
    //5436. 一维数组的动态和
    vector<int> runningSum(vector<int>& nums) {
        int sum = 0;
        vector<int> res;
        for (auto i : nums) {
            sum += i;
            res.push_back(sum);
        }
        return res;
    }
    //5437. 不同整数的最少数目
    //给你一个整数数组 arr 和一个整数 k 。现需要从数组中恰好移除 k 个元素，请找出移除后数组中不同整数的最少数目。
    int findLeastNumOfUniqueInts(vector<int>& arr, int k) {
        unordered_map<int, int> map1;
        for (auto i : arr) {
            if (map1.count(i)) {
                map1[i]++;
            } else {
                map1[i] = 1;
            }
        }
        map<int, vector<int>> newMap;
        for (auto i : map1) {
            if (newMap.count(i.second)) {
                newMap[i.second].push_back(i.first);
            } else {
                newMap[i.second] = {i.first};
            }
        }
        int allCount = map1.size();
        for (auto i : newMap) {
            if (k > 0) {
                int t = i.first * i.second.size();
                if (t > k) {
                    allCount -= k / i.first;
                    k -= k;
                } else {
                    k -= t;
                    allCount -= i.second.size();
                }
            } else {
                break;
            }
        }
        return allCount;
    }

    //5438. 制作 m 束花所需的最少天数
    /*给你一个整数数组 bloomDay，以及两个整数 m 和 k 。

    现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。

    花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好 可以用于 一束 花中。

    请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。
     */
    //not done
    int minDays(vector<int>& bloomDay, int m, int k) {
        if (m * k > bloomDay.size()) {
            return -1;
        }
        if (m * k == bloomDay.size()) {
            return bloomDay.size();
        }
        vector<int> days(bloomDay.size(),0);
        for (int i = 0; i < bloomDay.size(); i++) {
            int l = i - (k - 1);
            int r = i + (k - 1);
            if (l < 0) {
                l = 0;
            }
            if (r > bloomDay.size() - 1) {
                r = bloomDay.size() - 1;
            }
            int day = INT_MAX;
            for (int j = 0; j < k; j++) {
                int tmpD = bloomDay[i];
                for (int p = l + j; p < k; p++) {
                    tmpD = max(bloomDay[p],tmpD);
                }
                day = min(day, tmpD);
            }
            days[i] = day;
        }
        return 1;
    }


    //6.21周赛
    int xorOperation(int n, int start) {
        int res = 0;
        for (int i = 0; i < n; i++) {
            int t = start + 2 * i;
            res ^= t;
        }
        return res;
    }

    vector<string> getFolderNames(vector<string>& names) {
        unordered_map<string, int> map;
        vector<string> res;
        for (int i = 0; i < names.size(); i++) {
            string name = names[i];
            if (map.count(name) == 0) {
                res.push_back(name);
                map[name] = 1;
            } else {
                string newN = name + '(' + to_string(map[name]) + ')';
                map[name]++;
                map[newN] = 1;
                res.push_back(newN);
            }
            if (name[name.size()-1] == ')') {
                int j = name.size()-1;
                while (name[j] != '(') {
                    j--;
                }
                string newName = name.substr(0,j);
                int count = stoi(name.substr(j+1,name.size()-1-j-1));
                if (map.count(newName) == 0) {
                    map[newName] = 1;
                } else {
                    map[newName] = count + 1;
                }
            }
        }
        return res;
    }

    vector<int> avoidFlood(vector<int>& rains) {
        vector<int> res(rains.size());
        unordered_map<int, int> map;
        int count = 0;
        for (int i = 0; i < rains.size(); i++) {
            if (rains[i] == 0) {
                if (int(map.size()) >= count) {
                    count--;
                }
            } else {
                res[i] = -1;
                if (map.count(rains[i]) == 0) {
                    map[rains[i]] = 1;
                } else {
                    map[rains[i]]++;
                    count++;
                    if (count > 0) {
                        return vector<int>{};
                    }
                }
            }
        }
        if (count > 0) {
            return vector<int>{};
        } else {
            for (int i = 0; i < res.size(); i++) {
                if (res[i] != -1) {
                    for (auto it : map) {
                        if (it.second == 1) {
                            continue;
                        }
                        res[i] = it.first;
                        map[it.first] = map[it.first] - 1;
                        break;
                    }
                    if (res[i] == 0) {
                        res[i] = 1;
                    }
                }
            }
        }
        return res;
    }

    //6.28周赛
    bool isPathCrossing(string path) {
        int x = 0;
        int y = 0;
        unordered_map<int, set<int>> map;
        map[0] = set<int>{0};
        return pathCrossing(path, 0, x, y, map);
    }

    bool pathCrossing(string path, int i, int x, int y, unordered_map<int, set<int>>& map) {
        if (i == path.size()) {
            return false;
        }
        char c = path[i];
        switch (c) {
            case 'W':
                x -= 1;
                break;
            case 'N':
                y += 1;
                break;
            case 'E':
                x += 1;
                break;
            case 'S':
                y -= 1;
                break;
            default:
                break;
        }
        if (map.count(x)) {
            if (map[x].count(y)) {
                return true;
            } else {
                map[x].insert(y);
            }
        } else {
            map[x] = set<int>{y};
        }
        return pathCrossing(path, i+1, x, y, map);
    }

    bool canArrange(vector<int>& arr, int k) {
        unordered_map<int, int> map;
        for (int i = 0; i < arr.size(); i++) {
            int num = arr[i];
            int n = (num % k + k) % k;
            int target = k - n;
            if (target == k) {
                target = 0;
            }
            if (map.count(target)) {
                map[target] -= 1;
                if (map[target] == 0) {
                    map.erase(target);
                }
            } else {
                map[n] = 1;
            }
        }
        int count = 0;
        for (auto i : map) {
            count += i.second;
        }
        return count == 0;
    }

    //7.5周赛
    bool canMakeArithmeticProgression(vector<int>& arr) {
        if (arr.size() < 3) {
            return true;
        }
        sort(arr.begin(), arr.end());
        int c = arr[1] - arr[0];
        for (int i = 2; i < arr.size(); i++) {
            if (arr[i] - arr[i-1] != c) {
                return false;
            }
        }
        return true;
    }

    //给你一个只包含 0 和 1 的 rows * columns 矩阵 mat ，请你返回有多少个 子矩形 的元素全部都是 1 。
    int numSubmat(vector<vector<int>>& mat) {
        int ans = 0;
        int m=mat.size(),n=mat[0].size();
        for(int i=0;i<m;i++){
            vector<int>tmp(n,0);
            for(int j=i;j<m;j++){
                int cnt=0;
                for(int k=0;k<n;k++){
                    tmp[k]+=mat[j][k];
                }
                for(int k=0;k<n;k++){
                    if(tmp[k]==j-i+1) cnt++;
                    else cnt=0;
                    ans+=cnt;
                }
            }
        }
        return ans;
    }

    //7.12周赛
    /*
    给你一个整数数组 nums 。

    如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，就可以认为这是一组 好数对 。

    返回好数对的数目。
     */
    int numIdenticalPairs(vector<int>& nums) {
        int res = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (nums[i] == nums[j]) {
                    res += 1;
                }
            }
        }
        return res;
    }

    /*
     给你一个二进制字符串 s（仅由 '0' 和 '1' 组成的字符串）。

     返回所有字符都为 1 的子字符串的数目。

     由于答案可能很大，请你将它对 10^9 + 7 取模后返回。
     */
    int numSub(string s) {
        long left = 0;
        long right = 0;
        int res = 0;
        while (right < s.size()) {
            if (s[right] == '0') {
                left++;
                right++;
            } else {
                while (s[right] == '1') {
                    right++;
                }
                res += (right - left) * (right - left + 1) / 2 % 1000000007;
                left = right;
            }
        }
        return res;
    }

    /*
     给你一个由 n 个节点（下标从 0 开始）组成的无向加权图，该图由一个描述边的列表组成，其中 edges[i] = [a, b] 表示连接节点 a 和 b 的一条无向边，且该边遍历成功的概率为 succProb[i] 。

     指定两个节点分别作为起点 start 和终点 end ，请你找出从起点到终点成功概率最大的路径，并返回其成功概率。

     如果不存在从 start 到 end 的路径，请 返回 0 。只要答案与标准答案的误差不超过 1e-5 ，就会被视作正确答案。
     */

    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end) {
        vector<vector<pair<double,int>>> graph (n,vector<pair<double,int>>());
        for (int i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            graph[e[0]].push_back({succProb[i],e[1]});
            graph[e[1]].push_back({succProb[i],e[0]});
        }
        vector<int> visited(n,0);
        priority_queue<pair<double,int>> q;
        q.push({1,start});
        while(!q.empty()) {
            auto p = q.top();
            q.pop();
            auto curProb = p.first;
            auto curPos = p.second;
            if (visited[curPos]) continue;
            visited[curPos] = 1;
            if (curPos == end) return curProb;
            for ( auto next : graph[curPos]){
                double nextProb = next.first;
                int nextPos = next.second;
                if (visited[nextPos]) continue;
                q.push({curProb*nextProb,nextPos});
            }
        }
        return 0;
    }

};
