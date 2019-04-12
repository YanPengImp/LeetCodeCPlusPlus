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

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    Solution s = Solution();
    vector<int> a;
    a.push_back(1);
    a.push_back(3);
    a.push_back(2);
    a.push_back(4);
//    std::cout << s.findWords(a)[0] << std::endl;
//    std::cout << s.findWords(a)[1] << std::endl;
    std::cout << s.intToRoman(679) << std::endl;
    return 0;
}
