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
    a.push_back(2);
    s.missingTwo(a);
//    s.shellSort(a,a.size());
    s.heapSort(a,a.size());
    s.verifyPostorder(a);
    s.permute(a);
    s.canPartitionKSubsets(a,3);
    s.exchange(a);
    s.permutation("abc");
    s.waysToChange(10);
    s.findMinFibonacciNumbers(19);
    s.maximumSwap(115);
//    s.test();
    vector<int> yyy = {1,10,2,9,3,8,4,7,5,6};
    vector<int> zzzz = {4,3,1,1,3,3,2};
    s.findLeastNumOfUniqueInts(zzzz,3);
    s.minDays(yyy,4,2);
    vector<string> words;
    words.push_back("this");
    words.push_back("must");
    words.push_back("an");
    words.push_back("exampleasfwaas");
    words.push_back("offf");
    words.push_back("bu");
    s.fullJustify(words, 16);
    s.recoverFromPreorder("1-2--3--4-5--6--7");
    s.simplifyPath("/../");

    vector<string> names = vector<string>{"kingston(0)","kingston","kingston"};
    s.getFolderNames(names);
    vector<int> vods = vector<int>{1,2,0,0,2,1};
    s.avoidFlood(vods);
    s.isPathCrossing("NESWW");
    vector<int> v = vector<int>{-4,-7,5,2,9,1,10,4,-8,-3};
    s.canArrange(v,3);
    s.findKthLargest(v,4);

    vector<int> aa;
    aa.push_back(1);
    aa.push_back(1);
    aa.push_back(1);

    vector<int> bb;
    bb.push_back(1);
    bb.push_back(1);
    bb.push_back(0);

    vector<int> cc;
    cc.push_back(1);
    cc.push_back(1);
    cc.push_back(0);

    vector<vector<int>> rrr = {aa,bb,cc};
    s.numSubmat(rrr);

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
    s.generateTrees(1);
    vector<int> aaaaa;
    aaaaa.push_back(1);
    aaaaa.push_back(2);
    aaaaa.push_back(3);
    aaaaa.push_back(4);
    aaaaa.push_back(5);
    aaaaa.push_back(6);
    aaaaa.push_back(7);
    aaaaa.push_back(8);
    s.shuffle(aaaaa,4);

    TreeNode *root = new TreeNode(2);
    root->left = new TreeNode(2);
    s.isValidBST(root);

    TreeNode *xx = new TreeNode(3);
    xx->left = new TreeNode(2);
    xx->left->left = new TreeNode(2);
    xx->left->right = new TreeNode(1);
    xx->right = new TreeNode(3);
    xx->right->left = new TreeNode(2);
    xx->right->right = new TreeNode(0);

    s.iterationPreOrder(xx);
    s.iterationInOrder(xx);
    s.iterationPostOrder(xx);
    s.pseudoPalindromicPaths(xx);
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

    vector<int> nums{4,5,0,5};
    s.subarraysDivByK(nums,5);
    s.permutation33("abc");
    s.minDistance("horse","ors");
    vector<int> hhh = vector<int>{1,3};
    vector<int> www = vector<int>{1};
    s.maxArea(5,4,hhh,www);
//    s.oddEvenList(node);
//    s.reorderList(node);
//    a.push_back(4);
//    std::cout << s.findWords(a)[0] << std::endl;
//    std::cout << s.findWords(a)[1] << std::endl;
//    std::cout << s.permute(a) << std::endl;
    return 0;
}
