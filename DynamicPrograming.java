Dynamic Programming

1. Solves 0/1 knapsack in bottom up dynamic programming
2. Edit Distance ---> https://leetcode.com/problems/edit-distance/
3. Word Break ---->  https://leetcode.com/problems/word-break/
4. Longest Increasing Subsequence ---> https://leetcode.com/problems/longest-increasing-subsequence/
5. Unique Binary Search Trees https://leetcode.com/problems/unique-binary-search-trees/
6. Max Subset Sum No Adjacent (Advanced to Kadanes)(Similar to House Robber - logic indentified from discuss of House Robber)
7. Minimum Path Sum  https://leetcode.com/problems/minimum-path-sum/
8. Coin Change    https://leetcode.com/problems/coin-change/
9. Coin Change 2 -Total Unique Ways To Make Change   https://leetcode.com/problems/coin-change-2/
10. Longest Common Subsequence   https://leetcode.com/problems/longest-common-subsequence/ 
11. Longest Common Substring
12. Jump Game II 
---------------------
13. Matrix Chain Multiplication - Tushar Roy
14. Partition problem
15. Rod Cutting
16. Egg drop problem https://leetcode.com/problems/super-egg-drop/discuss/158974/C%2B%2BJavaPython-2D-and-1D-DP-O(KlogN)
17. Dungeon Game
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
1. Solves 0/1 knapsack in bottom up dynamic programming

    public int bottomUpDP(int val[], int wt[], int W){
	
        int K[][] = new int[val.length+1][W+1];
        for(int i=0; i <= val.length; i++){
            for(int j=0; j <= W; j++){
                if(i == 0 || j == 0){
                    K[i][j] = 0;
                    continue;
                }
                if(j - wt[i-1] >= 0){
                    K[i][j] = Math.max(K[i-1][j], K[i-1][j-wt[i-1]] + val[i-1]);    // if we want pick the current item 
                }else{
                    K[i][j] = K[i-1][j];     // if we cant pick the current Item.
                }
            }
        }
        return K[val.length][W];
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

2. Edit Distance ---> https://leetcode.com/problems/edit-distance/

    public int minDistance(String word1, String word2) {
	
        int m = word1.length();
        int n = word2.length();
        
        int[][] cost = new int[m+1][n+1];
        
        for(int i=0; i<=m; i++){
            for(int j=0; j<=n; j++){
                
                if(i==0){ cost[i][j] = j; }
                
                else if(j==0){ cost[i][j] = i; }
                
                else if(word1.charAt(i-1) == word2.charAt(j-1)){
                    cost[i][j] = cost[i-1][j-1];
                }
                
                else
                    cost[i][j] = 1 + min(cost[i-1][j-1],cost[i][j-1],cost[i-1][j]);
            }
        }
        
        return cost[m][n];
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

3. Word Break ---->  https://leetcode.com/problems/word-break/

    public boolean wordBreak(String s, List<String> wordDict) {
        
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        
        for (int lo = 1; lo <= n; lo++) {
		
            if (!dp[lo - 1]) continue;

            for (String word : wordDict) {
                int hi = lo - 1 + word.length();
				
                if (hi <= n && s.substring(lo - 1, hi).equals(word)) {
                    dp[hi] = true;
                    
                }
				
            }
        }
        
        return dp[n];
        
    }

*--------------------------------------------------------*-----------------------------------------------------------------------------------*

4. Longest Increasing Subsequence ---> https://leetcode.com/problems/longest-increasing-subsequence/

    public int lengthOfLIS(int[] nums) {
        
        if(nums.length == 0 || nums.length == 1) return nums.length;
        
        int[] answer = new int[nums.length];
        
        for(int i=0; i<nums.length; i++){
            answer[i] = 1;
        }
        
        int maxLen = 1;
		
        for(int i=1; i<nums.length; i++){
		
            for(int j=0; j<i; j++){
			
                if(nums[i] > nums[j] && answer[i] < answer[j] + 1){
                    answer[i] = answer[j] + 1;
                    maxLen = Math.max(maxLen, answer[i]);
                }
				
            }
			
        }
        
        return maxLen;
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
5. Unique Binary Search Trees https://leetcode.com/problems/unique-binary-search-trees/

    public int numTrees(int n) {
        
        if(n==0 || n==1) return n;
        
        int[] answer =new int[n+1];
        answer[0] = 1;
        answer[1] = 1;
        
        for(int i=2; i<=n; i++){
            for(int j=1; j<=i; j++){
                answer[i] += answer[i-j] * answer[j-1];
            }
        }
        
        return answer[n];
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

6. Max Subset Sum No Adjacent (Advanced to Kadanes)(Similar to House Robber - logic indentified from discuss of House Robber)

  public static int maxSubsetSumNoAdjacent(int[] array) {
  
		if(array.length == 0) return 0;
		if(array.length == 1) return array[0];
		int incl = 0, notIncl = 0, prevIncl = 0, prevNotIncl = 0;
		
		for(int i=0; i<array.length; i++){
			incl = prevNotIncl + array[i];
			notIncl = Math.max(prevIncl, prevNotIncl);
			
			prevIncl = incl;
			prevNotIncl = notIncl;
		}
		
    
        return Math.max(incl, prevNotIncl);
  }
  
   ----------------------------------------------------------
   
  //O(n) and O(n) Space
  public static int maxSubsetSumNoAdjacent(int[] array) {
  
		if(array.length == 0) return 0;
		if(array.length == 1) return array[0];
		
		int[] maxSums = new int[array.length];
		
		maxSums[0] = array[0];
		maxSums[1] = Math.max(maxSums[0], maxSums[1]);
		
		for(int i=2; i<array.length; i++){
			maxSums[i] = Math.max(maxSums[i-1], maxSums[i-2] + array[i]);
		}
		
    
        return maxSums[array.length-1];
  }
  
  ----------------------------------------------------------
  House Robber II
  
    public int rob(int[] nums) {
      int n = nums.length;
      if(n == 0) return 0;
      if(n == 1) return nums[0];
     
	  return Math.max(rob(nums, 0, n-2), rob(nums, 1, n-1));
    }

    public int rob(int[] nums, int lo, int hi) {
	
        int preRob = 0, preNotRob = 0, rob = 0, notRob = 0;
		
        for (int i = lo; i <= hi; i++) {
		
            rob = preNotRob + nums[i];
            notRob = Math.max(preRob, preNotRob);

            preNotRob = notRob;
            preRob = rob;
        }
        return Math.max(rob, notRob);
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

7. Minimum Path Sum  https://leetcode.com/problems/minimum-path-sum/

    public int minPathSum(int[][] grid) {
        
        if(grid.length == 0) return 0;
        
        int row = grid.length;
        int col = grid[0].length;
        
        int[][] answer = new int[row][col];
        answer[0][0] = grid[0][0];
        
        for(int i=1; i<col; i++){
            answer[0][i] = answer[0][i-1] + grid[0][i];
        }
        
        for(int i=1; i<row; i++){
            answer[i][0] = answer[i-1][0] + grid[i][0];
        }
        
        for(int i=1; i<row; i++){
            for(int j=1; j<col; j++){
              answer[i][j] = grid[i][j] + Math.min(answer[i-1][j], answer[i][j-1]);
                                
            }
        }
        
        return answer[row-1][col-1];   
        
    }
  
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
8. Coin Change    https://leetcode.com/problems/coin-change/


  public int coinChange(int[] coins, int amount) {
    int max = amount + 1;
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, max);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
      for (int j = 0; j < coins.length; j++) {
        if (coins[j] <= i) {
          dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
        }
      }
    }
    return dp[amount] > amount ? -1 : dp[amount];
  }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
9. Coin Change 2 -Total Unique Ways To Make Change   https://leetcode.com/problems/coin-change-2/

    public int change(int amount, int[] coins) {
        
        int row = coins.length+1;
        int col = amount+1;
        int[][] dp = new int[row][col];
        
        dp[0][0] = 1;
        
        for(int i=1; i<row; i++)
            dp[i][0] = 1;
        
        for(int i=1; i<col; i++)
            dp[0][i] = 0;
        
        for(int i=1; i<row; i++){
            for(int j=1; j<col; j++){
                if(j>= coins[i-1])
                    dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]];
                else
                    dp[i][j] = dp[i-1][j];
            }
        }     
        
        return dp[row-1][col-1];
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

10. Longest Common Subsequence   https://leetcode.com/problems/longest-common-subsequence/  

    public int longestCommonSubsequence(String text1, String text2) {
        
        if(text1 == null || text2==null || text1.length() == 0 || text2.length() == 0) 
            return 0;
        
        int n = text1.length();
        int m = text2.length();
        
        int[][] dp =new int[n+1][m+1];

        for(int i=0; i<n+1; i++){
            dp[i][0] = 0;
        }
        
        for(int i=0; i<m+1; i++){
            dp[0][i] = 0;
        }
       
        for(int i=1; i<n+1; i++){
            for(int j=1; j<m+1; j++){
                if(text1.charAt(i-1) == text2.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
        
        return dp[n][m];
        
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
11. Longest Common Substring

    public int longestCommonSubstring(char str1[], char str2[]){
        int T[][] = new int[str1.length+1][str2.length+1];
        
        int max = 0;
        for(int i=1; i <= str1.length; i++){
            for(int j=1; j <= str2.length; j++){
                if(str1[i-1] == str2[j-1]){
                    T[i][j] = T[i-1][j-1] +1;
                    if(max < T[i][j]){
                        max = T[i][j];
                    }
                }
            }
        }
        return max;
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
12. Jump Game II   ----------------> nums.length-1 in for loop (Important point to be noted)

    public int jump(int[] nums) {
        
        int currentMax = 0, prevMax = 0, jumps = 0;
        
        for(int i=0; i<nums.length-1; i++){
            currentMax = Math.max(currentMax, i+nums[i]);
            if(i==prevMax){
                jumps++;
                prevMax = currentMax;
            }
        }
        
        return jumps;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

13. Matrix Chain Multiplication - Tushar Roy

public class MatrixMultiplicationCost {

    public int findCost(int arr[]){
        int[][] m = new int[arr.length][arr.length];
        int q = 0;
		int i, j, k, L, q;
		
        // cost is zero when multiplying one matrix. 
        for (i = 1; i < n; i++) 
            m[i][i] = 0; 
			
        // L is chain length. 
        for (L=2; L<n; L++) 
        { 
            for (i=1; i<n-L+1; i++) 
            { 
                j = i+L-1; 
                if(j == n) continue; 
                m[i][j] = Integer.MAX_VALUE; 
                for (k=i; k<=j-1; k++) 
                { 
                    // q = cost/scalar multiplications 
                    q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]; 
                    if (q < m[i][j]) 
                        m[i][j] = q; 
                } 
            } 
        } 
  
        return m[1][n-1]; 
    }
    
    public static void main(String args[]){
        MatrixMultiplicationCost mmc = new MatrixMultiplicationCost();
        int arr[] = {4,2,3,5,3};
        int cost = mmc.findCost(arr);
        System.out.print(cost);
    }
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
14. Partition problem

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
15. Rod Cutting

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
16. Egg drop problem https://leetcode.com/problems/super-egg-drop/discuss/158974/C%2B%2BJavaPython-2D-and-1D-DP-O(KlogN)

The dp equation is:
	dp[m][k] = dp[m - 1][k - 1] + dp[m - 1][k] + 1,
	which means we take 1 move to a floor,
		if egg breaks, then we can check dp[m - 1][k - 1] floors.
		if egg doesn't breaks, then we can check dp[m - 1][k] floors.

    public int superEggDrop(int K, int N) {
	
        int[][] dp = new int[N + 1][K + 1];
        int m = 0;
        while (dp[m][K] < N) {
            ++m;
            for (int k = 1; k <= K; ++k)
                dp[m][k] = dp[m - 1][k - 1] + dp[m - 1][k] + 1;
        }
        return m;
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
17. Dungeon Game

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
