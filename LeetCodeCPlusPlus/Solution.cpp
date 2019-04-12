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
#include <queue>

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

    //50.计算x的n次幂
    double myPow(double x, int n) {
        if (n == 0) return 1;
        double half = myPow(x, n / 2);
        if (n % 2 == 0) return half * half;
        else if (n > 0) return half * half * x;
        else return half * half / x;
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

        //找到后比如 0-len表示最长的回文，len-length()-1就是没有匹配上的，反转加在最前面就是
        for(i=(int)s.length()-1;i>len;i--)
            result+=s[i];

        result+=s;
        return result;
    }

    //237.删除链表中节点
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
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
};

