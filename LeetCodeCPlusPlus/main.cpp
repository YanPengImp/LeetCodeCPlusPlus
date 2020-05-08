//
//  main.cpp
//  LeetCodeCPlusPlus
//
//  Created by Imp on 2019/1/9.
//  Copyright © 2019 imp. All rights reserved.
//

#include <iostream>
#include "Solution.cpp"

using namespace std;

int n = 3;
int a[6], b[6];

void print() {
    for (int i = 1; i <= n; i++) {
        std::cout<< a[i];
    }
    std::cout<<endl;
}

void dfs(int i) {
    if (i == n+1) {
        print();
        return;
    }
    for (int j = 1; j <= n; j++) {
        if (b[j] == 0) {
            a[i] = j;
            b[j] = 1;
            dfs(i+1);
            b[j] = 0;
        }
    }
}

int main(int argc, const char * argv[]) {
    // insert code here...
    dfs(1);

    std::cout << "Hello, World!\n";
    Solution s = Solution();
    vector<int> a;
    a.push_back(1);
    a.push_back(6);
    a.push_back(3);
    a.push_back(2);
    a.push_back(5);
    s.shellSort(a,a.size());
    s.verifyPostorder(a);
    s.permute(a);
    s.canPartitionKSubsets(a,3);
    s.exchange(a);
    s.permutation("abc");


    vector<string> words;
    words.push_back("this");
    words.push_back("must");
    words.push_back("an");
    words.push_back("exampleasfwaas");
    words.push_back("offf");
    words.push_back("bu");
    s.fullJustify(words, 16);

    s.simplifyPath("/../");

    vector<int> aa;
    aa.push_back(1);
    aa.push_back(1);
    aa.push_back(3);

    vector<int> bb;
    bb.push_back(4);
    bb.push_back(3);
    bb.push_back(2);

    vector<int> cc;
    cc.push_back(1);
    cc.push_back(3);
    cc.push_back(2);

//    vector<int> dd;
//    dd.push_back(0);
//    dd.push_back(0);
//    dd.push_back(0);

    vector<vector<int>> tt;
    tt.push_back(aa);
    tt.push_back(bb);
    tt.push_back(cc);
    s.maxValue(tt);
//    s.rotate(tt);

    ListNode *node = new ListNode(1);
    node->next = new ListNode(1);
    node->next->next = new ListNode(2);
    node->next->next->next = new ListNode(1);
    node->next->next->next->next = new ListNode(1);
//    node->next->next->next->next = new ListNode(5);
//    node->next->next->next->next->next = new ListNode(6);
    ListNode *nnnode = NULL;
//    s.splitListToParts(nnnode,4);
//    s.sortedListToBST(node);
    s.numDecodings("226");
    s.generateTrees(3);
    vector<int> aaaaa;
    aaaaa.push_back(3);
    aaaaa.push_back(1);
    aaaaa.push_back(0);
    aaaaa.push_back(3);
    aaaaa.push_back(5);
    aaaaa.push_back(8);
    aaaaa.push_back(11);
    aaaaa.push_back(13);
    aaaaa.push_back(14);

    TreeNode *xx = new TreeNode(-2);
//    xx->left = new TreeNode(2);
    xx->right = new TreeNode(-3);
    s.recoverTree(xx);
    s.rob(aaaaa);

    s.letterCombinations("23");
    s.solveNQueens(4);
    s.reverseWords(" this   is the sky  ");
    s.pathSum(xx,-5);

    s.isPalindrome(node);

    s.headCreateListNode(aaaaa);
    s.tailCreateListNode(aaaaa);
    s.findNthDigit(18);
    s.translateNum(12258);
    s.firstUniqChar("aadadaad");
//    s.oddEvenList(node);
//    s.reorderList(node);
//    a.push_back(4);
//    std::cout << s.findWords(a)[0] << std::endl;
//    std::cout << s.findWords(a)[1] << std::endl;
//    std::cout << s.permute(a) << std::endl;
    return 0;
}
