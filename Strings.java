
/*
1. 443. String Compression
2. 91. Decode Ways  https://leetcode.com/problems/decode-ways/
3. 76. Minimum Window Substring
4. Longest Palindromic Substring 
5. Generate all Parentheses II
6. 60. Kth Permutation Sequence
7. 131. Palindrome Partitioning
8. 394. Decode String

*/
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
1. 443. String Compression

Input: ["a","a","b","b","c","c","c"]
Output:6
Explanation: the first 6 characters of the input array should be: ["a","2","b","2","c","3"]

*/
    public int compress(char[] chars) {
        int indexAns = 0, index = 0;
        while(index < chars.length){
            char currentChar = chars[index];
            int count = 0;
            while(index < chars.length && chars[index] == currentChar){
                index++;
                count++;
            }
            chars[indexAns++] = currentChar;
            if(count != 1)
                for(char c : Integer.toString(count).toCharArray()) 
                    chars[indexAns++] = c;
        }
        return indexAns;
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
2. 91. Decode Ways  https://leetcode.com/problems/decode-ways/

Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).

*/
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0; // important point - need to be noted 
        for (int i = 2; i <= n; i++) {
            int first = Integer.valueOf(s.substring(i - 1, i));
            int second = Integer.valueOf(s.substring(i - 2, i));
            if (first >= 1 && first <= 9) {
               dp[i] += dp[i-1];  
            }
            if (second >= 10 && second <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[n];
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
3. 76. Minimum Window Substring

Input :  S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"

*/
    public String minWindow(String s, String t) {
		
        int[] map = new int[128];
		
        for (char c : t.toCharArray()) {
            map[c] += 1;
        }
		
        int begin = 0;
        int len = Integer.MAX_VALUE;
        int count = t.length();
		
        for (int left=0, right=0; right<s.length(); right++) {
            char c = s.charAt(right);
            map[c]--;
            if(map[c]>=0) count--; 
			
            while (count == 0) {
                char lc = s.charAt(left);
                map[lc]++;
                if (map[lc]>0) {
                    if (right-left+1<len) {
                        begin = left;
                        len = right-left+1;
                    }
                    count++;
                }
                left++;
            }
			
        }
		
        return len==Integer.MAX_VALUE?"":s.substring(begin, begin+len);
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
4. Longest Palindromic Substring
 
*/
    public String longestPalindrome(String A) {
        
        char[] input = A.toCharArray();
        
        int left = 0, right = 0, len = 0;
        
        for(int i=0; i<input.length; i++){
            
            int len1 = isPalin(input, i, i);
            int len2 = isPalin(input, i, i+1);
            int maxLen = Math.max(len1, len2);

            if(len < maxLen){
                left = i - (maxLen-1)/2;
                len = maxLen;
            }
        }
        
        return len == 0 ? "" : A.substring(left, left+len);
        
    }
    
    public int isPalin(char[] B, int left, int right){
        
        while(left>=0 && right < B.length && B[left] == B[right]){
            left--;
            right++;
        }
        
        return right - left - 1;  // used +1 instead of -1, could not identify , when the while loop ends left and right will be in out of position
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
5. Generate all Parentheses II

*/
    public ArrayList<String> generateParenthesis(int A) {
        
        ArrayList<String> result = new ArrayList<String>();
        generateParenthesisUtil(0, 0, "",result, A);
        return result;
    }
    
    public void generateParenthesisUtil(int open, int close, String cur, ArrayList<String> result, int A){
                                            
        if(cur.length() == 2*A){
            result.add(cur);
            return;
        }

        if(open<A) 
          generateParenthesisUtil(open+1, close, cur + "(", result, A);
                                   
        if(close<open)  // close should be less than open in If condition .. missed this point.
          generateParenthesisUtil(open, close+1, cur + ")", result, A);
                                            
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
/*
6. 60. Kth Permutation Sequence

https://leetcode.com/problems/permutation-sequence/discuss/22507/%22Explain-like-I'm-five%22-Java-Solution-in-O(n)

I'm sure somewhere can be simplified so it'd be nice if anyone can let me know. The pattern was that:

say n = 4, you have {1, 2, 3, 4}

If you were to list out all the permutations you have

1 + (permutations of 2, 3, 4)

2 + (permutations of 1, 3, 4)

3 + (permutations of 1, 2, 4)

4 + (permutations of 1, 2, 3)


We know how to calculate the number of permutations of n numbers... n! So each of those with permutations of 3 numbers means there are 6 possible permutations. Meaning there would be a total of 24 permutations in this particular one. So if you were to look for the (k = 14) 14th permutation, it would be in the

3 + (permutations of 1, 2, 4) subset.

To programmatically get that, you take k = 13 (subtract 1 because of things always starting at 0) and divide that by the 6 we got from the factorial, which would give you the index of the number you want. In the array {1, 2, 3, 4}, k/(n-1)! = 13/(4-1)! = 13/3! = 13/6 = 2. The array {1, 2, 3, 4} has a value of 3 at index 2. So the first number is a 3.

Then the problem repeats with less numbers.

The permutations of {1, 2, 4} would be:

1 + (permutations of 2, 4)

2 + (permutations of 1, 4)

4 + (permutations of 1, 2)

But our k is no longer the 14th, because in the previous step, we've already eliminated the 12 4-number permutations starting with 1 and 2. So you subtract 12 from k.. which gives you 1. Programmatically that would be...

k = k - (index from previous) * (n-1)! = k - 2*(n-1)! = 13 - 2*(3)! = 1

In this second step, permutations of 2 numbers has only 2 possibilities, meaning each of the three permutations listed above a has two possibilities, giving a total of 6. We're looking for the first one, so that would be in the 1 + (permutations of 2, 4) subset.

Meaning: index to get number from is k / (n - 2)! = 1 / (4-2)! = 1 / 2! = 0.. from {1, 2, 4}, index 0 is 1


so the numbers we have so far is 3, 1... and then repeating without explanations.


{2, 4}

k = k - (index from pervious) * (n-2)! = k - 0 * (n - 2)! = 1 - 0 = 1;

third number's index = k / (n - 3)! = 1 / (4-3)! = 1/ 1! = 1... from {2, 4}, index 1 has 4

Third number is 4


{2}

k = k - (index from pervious) * (n - 3)! = k - 1 * (4 - 3)! = 1 - 1 = 0;

third number's index = k / (n - 4)! = 0 / (4-4)! = 0/ 1 = 0... from {2}, index 0 has 2

Fourth number is 2


Giving us 3142. If you manually list out the permutations using DFS method, it would be 3142. Done! It really was all about pattern finding.

*/

public String getPermutation(int n, int k) {
    int pos = 0;
    List<Integer> numbers = new ArrayList<>();
    int[] factorial = new int[n+1];
    StringBuilder sb = new StringBuilder();
    
    // create an array of factorial lookup
    int sum = 1;
    factorial[0] = 1;
    for(int i=1; i<=n; i++){
        sum *= i;
        factorial[i] = sum;
    }
    // factorial[] = {1, 1, 2, 6, 24, ... n!}
    
    // create a list of numbers to get indices
    for(int i=1; i<=n; i++){
        numbers.add(i);
    }
    // numbers = {1, 2, 3, 4}
    
    k--; // subtract 1 because of things always starting at 0
    
    for(int i = 1; i <= n; i++){
        int index = k/factorial[n-i];
        sb.append(String.valueOf(numbers.get(index)));
        numbers.remove(index);
        k-=index*factorial[n-i];
    }
    
    return String.valueOf(sb);
}


*--------------------------------------------------------*--------------------------------------------------------------------------------------

/*

7. 131. Palindrome Partitioning

Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:

Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
*/

class Solution {
    
    List<List<String>> result;
    public List<List<String>> partition(String s) {
        result = new ArrayList<List<String>>();
        partitionUtil(s, 0, new ArrayList<String>());
        return result;
    }
    
    private void partitionUtil(String a, int start, ArrayList<String> current){
        if(start == a.length()){
            result.add(new ArrayList<String>(current));
            return;
        }
        
        for(int i=start; i<a.length(); i++){
           if(isPalin(a, start, i)){
               String currentString = a.substring(start, i+1);
               current.add(currentString);
               partitionUtil(a, i+1, current);
               current.remove(current.size()-1);
           } 
        }
    }
    
    private boolean isPalin(String a, int left, int right){
        while(left<right){
            if(a.charAt(left) != a.charAt(right)) return false;
            else{
                left++;
                right--;
            }
        }
        
        return true;
    }
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------

/*
8. 394. Decode String

Input: s = "3[a]2[bc]"
Output: "aaabcbc"

*/ 

class Solution {
    public String decodeString(String s) {
        
        Stack<Integer> count = new Stack<>();
        Stack<String> str = new Stack<>();
        String res =  "";
        
        int i=0;
        char[] input = s.toCharArray();
        
        while(i<s.length()){
            if(Character.isDigit(input[i])){
                int val = 0;
                while(Character.isDigit(input[i])){
                    val = (val * 10) + Integer.valueOf(input[i] - '0');
                    i++;
                }
                count.push(val);
            }else if(input[i] == '['){
                str.push(res);
                res = "";
                i++;
            }else if(input[i] == ']'){
                int times = count.pop();
                StringBuilder curr = new StringBuilder(str.pop());
                for(int j=0; j<times; j++){
                    curr.append(res);
                }
                res = curr.toString();
                i++;
                
            }else{
                res += input[i++];
            }
        }
        
        return res;
    }
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------
