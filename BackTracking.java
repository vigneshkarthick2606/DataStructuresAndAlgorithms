	
1. 93. Restore IP Addresses (BackTracking)
2. Next greater element in same order as input (Stack)
3. Combinations  (BackTracking)
4. Combination Sum  (BackTracking)
5. Combination Sum II (BackTracking)
6. Subset (BackTracking)
7. Subsets II (BackTracking)
8. N-Queens (BackTracking)
9. Sudoku Solver (BackTracking)
10. Permutations (BackTracking)
11. Palindrome Decompositions (BackTracking)
12. Letter Combinations of a Phone Number (BackTracking)
13. Rat in a Maze (BackTracking)

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
1. 93. Restore IP Addresses  https://leetcode.com/problems/restore-ip-addresses/

public List<String> restoreIpAddresses(String rawIpString) {
    List<String> restoredIps = new ArrayList<>();
    restoreIps(0, 0, new int[4], rawIpString, restoredIps);

    return restoredIps;
  }

  private void restoreIps(int progressIndex, int currentSegment, int[] ipAddressSegments, String rawIpString, List<String> restoredIps){
    /*
      If we have filled 4 segments (0, 1, 2, 3) and we are on the 4th,
      we will only record an answer if the string was decomposed fully
    */
    if (currentSegment == 4 && progressIndex == rawIpString.length()) {
      restoredIps.add(buildIpStringFromSegments(ipAddressSegments));
    } else if (currentSegment == 4) {
      return;
    }

    /*
      Generate segments to try, a segment can be 1 to 3 digits long.
    */
    for (int segLength = 1; segLength <= 3 && progressIndex + segLength <= rawIpString.length(); segLength++) {

      // Calculate 1 index past where the segment ends index-wise in the original raw ip string
      int onePastSegmentEnd = progressIndex + segLength;

      // Extract int value from our snapshot from the raw ip string
      String segmentAsString = rawIpString.substring(progressIndex, onePastSegmentEnd);
      int segmentValue = Integer.parseInt(segmentAsString);

      // Check the "snapshot's" validity - if invalid break iteration
      if (segmentValue > 255 || segLength >= 2  && segmentAsString.charAt(0) == '0') {
        break;
      }

      // Add the extracted segment to the working segments
      ipAddressSegments[currentSegment] = segmentValue;

      // Recurse on the segment choice - when finished & we come back here, the next loop iteration will try another segment
      restoreIps(progressIndex + segLength, currentSegment + 1, ipAddressSegments, rawIpString, restoredIps);
    }
  }

  private String buildIpStringFromSegments(int[] ipAddressSegments) {
    return ipAddressSegments[0] + "." + ipAddressSegments[1] + "."+ ipAddressSegments[2] + "." + ipAddressSegments[3];
  }
  
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

	public List<String> restoreIpAddresses(String s) {
		List<String> ret = new LinkedList<>();
		int[] path = new int[4];
		helper(ret, s, 0,  path, 0);
		return ret;
	}

	void helper(List<String> acc, String s, int idx, int[] path,  int segment){
		
		if(segment == 4 && idx == s.length() ){
			acc.add(path[0] + "." + path[1] + "."+ path[2] + "." + path[3]);
			return ;
		}else if(segment == 4 || idx == s.length() ){
			return ;
		}
		//Generate segments to try, a segment can be 1 to 3 digits long.
		for(int len = 1; len <= 3 && idx + len <= s.length() ; len ++){
			int val = Integer.parseInt(s.substring(idx, idx + len));
			// range check, no leading 0.
			if(val > 255 || len >= 2  && s.charAt(idx) == '0') 
				break; 
				
			path[segment] = val;
			helper(acc, s, idx + len, path, segment + 1);
			path[segment] = -1; // for debug. 
		}
	}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

2. Next greater element in same order as input

Input : [4, 5, 2, 25]
Output : 5 25 25 -1

    public ArrayList<Integer> nextGreater(ArrayList<Integer> A) {
        int n = A.size();
        ArrayList<Integer> result = new ArrayList<Integer>();
        
        Stack<Integer> stk = new Stack<Integer>();
        
        for(int a: A)
           result.add(-1);
        
        for(int i=n-1; i>=0; i--){   // loop from backwards --> always remember this 
            
            while(!stk.isEmpty() && stk.peek() <= A.get(i))
                stk.pop();
            
              
            if(!stk.isEmpty())
               result.set(i, stk.peek());
               
            stk.push(A.get(i));
            
        }
        
        return result;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
3. Combinations 

Given two integers n and k, return all possible combinations of k numbers out of 1 2 3 ... n.

    public ArrayList<ArrayList<Integer>> combine(int A, int B) {
        
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> current = new ArrayList<Integer>();
        if(A<=0 || B > A) return result;
        
        dfs(A, 1, B, current, result);
        
        return result;
    }
    
    public void dfs(int N, int start, int K, ArrayList<Integer> current, ArrayList<ArrayList<Integer>> result){
        
        if(current.size() == K){
            result.add(new ArrayList<Integer>(current));
            return;
        }
        
        for(int i=start; i<=N; i++){
            current.add(i);
            dfs(N, i+1, K, current, result);
            current.remove(current.size()-1);
        }
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
4. Combination Sum

Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

    public ArrayList<ArrayList<Integer>> combinationSum(ArrayList<Integer> A, int B) {
        
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> current = new ArrayList<Integer>();
        
        if(A==null || A.size()==0) return result;
        Collections.sort(A);
        dfs(A, 0, B, current, result);
        
        return result;
    }
    
    public void dfs(ArrayList<Integer> A, int start, int K, ArrayList<Integer> current, ArrayList<ArrayList<Integer>> result){
        if(K<0)
          return;
        
        if(K == 0){
            result.add(new ArrayList<Integer>(current));
            return;
        }
        
        for(int i=start; i<A.size(); i++){
            if(i>0 && A.get(i) == A.get(i-1)) continue;    // important part
            current.add(A.get(i));
            dfs(A, i, K-A.get(i), current, result);
            current.remove(current.size()-1);
        }
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
5. Combination Sum II

Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

    public ArrayList<ArrayList<Integer>> combinationSum(ArrayList<Integer> a, int b) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> current = new ArrayList<Integer>();
        
        if(a==null || a.size()==0) return result;
        Collections.sort(a);
        dfs(a, 0, b, current, result);
        
        return result;
    }
    
    public void dfs(ArrayList<Integer> A, int start, int K, ArrayList<Integer> current, ArrayList<ArrayList<Integer>> result){
        if(K<0)
          return;
        
        if(K == 0){
            result.add(new ArrayList<Integer>(current));
            return;
        }
        
        for(int i=start; i<A.size(); i++){
            if(i!=start && A.get(i) == A.get(i-1)) continue;   // 
            current.add(A.get(i));
            dfs(A, i+1, K-A.get(i), current, result);
            current.remove(current.size()-1);
        }
        
    } 

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
6. Subset

Given a set of distinct integers, S, return all possible subsets.
	Elements in a subset must be in non-descending order.
	The solution set must not contain duplicate subsets.
	Also, the subsets should be sorted in ascending ( lexicographic ) order.
	The list is not necessarily sorted.


Order of the output is different for Iterative and recursive
------------------------------------------------------------------
Iterative method can be used if only subsets are asked and no need to sort the list initially.


Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]---> ordering matters in some situations
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> subsets= new ArrayList<List<Integer>>();
        subsets.add(new ArrayList<Integer>());
        
        for(int i=0; i<nums.length; i++){
            int len= subsets.size();
            for(int j=0; j<len; j++){
               List<Integer> current= new ArrayList<Integer>(subsets.get(j));
               current.add(nums[i]);
               subsets.add(current);
            }
            
        }
        
        return subsets;
    }

-----------------------------------------------------------------------

Output: [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]] ---> ordering matters in some situations

    public ArrayList<ArrayList<Integer>> subsets(ArrayList<Integer> A) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        Collections.sort(A);
        backtrack(list, new ArrayList<>(), A, 0);
        return list;
   }

    private void backtrack(ArrayList<ArrayList<Integer>> list , ArrayList<Integer> tempList, ArrayList<Integer> A, int start){
        list.add(new ArrayList<>(tempList));
        for(int i = start; i < A.size(); i++){
            tempList.add(A.get(i));
            backtrack(list, tempList, A, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
7. Subsets II

Given a collection of integers that might contain duplicates, S, return all possible subsets.

	Elements in a subset must be in non-descending order.
	The solution set must not contain duplicate subsets.
	The subsets must be sorted lexicographically.


    public ArrayList<ArrayList<Integer>> subsetsWithDup(ArrayList<Integer> A) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        Collections.sort(A);
        backtrack(list, new ArrayList<>(), A, 0);
        return list;
   }

    private void backtrack(ArrayList<ArrayList<Integer>> list , ArrayList<Integer> tempList, ArrayList<Integer> A, int start){
        list.add(new ArrayList<>(tempList));
        for(int i = start; i < A.size(); i++){
            if(i!=start && A.get(i)==A.get(i-1)) continue; // important condition to point out, compared with start as the first time alone it is valid
            tempList.add(A.get(i));
            backtrack(list, tempList, A, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }


*--------------------------------------------------------*--------------------------------------------------------------------------------------*
8. N-Queens 

OutPut: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]

    List<List<String>> result = new ArrayList<>();
    
    public List<List<String>> solveNQueens(int n) {
        boolean[] visited = new boolean[n];
        boolean[] dia1 = new boolean[2*n-1];
        boolean[] dia2 = new boolean[2*n-1];
        
        dfs(n, new ArrayList<String>(),visited,dia1,dia2,0);
        
        return result;
    }
    
    private void dfs(int n,List<String> list,boolean[] visited,boolean[] dia1,boolean[] dia2,int rowIndex){
		
        if(rowIndex == n){
            result.add(new ArrayList<String>(list));
            return;
        }
        
        for(int i=0;i<n;i++){
			//verify if current position is valid or not
            if(visited[i] || dia1[rowIndex+i] || dia2[rowIndex-i+n-1])
                continue;
            
			//create the current string  
            char[] charArray = new char[n];
            Arrays.fill(charArray,'.');
            charArray[i] = 'Q';
            String stringArray = new String(charArray);
            
			//place the current and call on recursion
            list.add(stringArray);
            visited[i] = true;
            dia1[rowIndex+i] = true;
            dia2[rowIndex-i+n-1] = true;

            dfs(n,list,visited,dia1,dia2,rowIndex+1);
            
			//if this point is reached the current entry is not valid, remove 
            list.remove(list.size()-1);
            visited[i] = false;
            dia1[rowIndex+i] = false;
            dia2[rowIndex-i+n-1] = false;
        }
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
9. Sudoku Solver

    public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0)
            return;
        solve(board);
    }
    
    public boolean solve(char[][] board){
	
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(board[i][j] == '.'){
                    for(char c = '1'; c <= '9'; c++){//trial. Try 1 through 9
                        if(isValid(board, i, j, c)){
                            board[i][j] = c; //Put c for this cell
                            
                            if(solve(board))
                                return true; //If it's the solution return true
                            else
                                board[i][j] = '.'; //Otherwise go back
                        }
                    }
                    
                    return false;
                }
            }
        }
        return true;
    }
	
    private boolean isValid(char[][] board, int row, int col, char c){
        int regionRow = 3 * (row / 3);    //region start row
        int regionCol = 3 * (col / 3);    //region start col
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == c) return false; //check row
            if (board[row][i] == c) return false; //check column
            if (board[regionRow + i / 3][regionCol + i % 3] == c) return false; //check 3*3 block
        }
        return true;
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
10. Permutations
    Given a collection of numbers, return all possible permutations.

    ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
    public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> A) {
        permute(new ArrayList<>(A), 0, A.size()-1);
        return result;
    }
    
    public void permute(ArrayList<Integer> current, int left, int right){
        if(left==right){
            result.add(new ArrayList<>(current));
            return;
        }
        
        for(int i=left; i<=right; i++){
            swap(current, left, i);
            permute(current, left+1, right);
            swap(current, left, i);
        }
    }
    
    public void swap(ArrayList<Integer> list, int i, int j){
        int temp = list.get(i);
        list.set(i, list.get(j));
        list.set(j, temp);
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
11. Palindrome Decompositions

For example, given s = "aab",
Return
  [
    ["a","a","b"]
    ["aa","b"],
  ]

public class Solution {
    
    ArrayList<ArrayList<String>> result; 
    public ArrayList<ArrayList<String>> partition(String a) {
        result = new ArrayList<ArrayList<String>>();
        partitionUtil(a, 0, new ArrayList<String>());
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
               partitionUtil(a, i+1, current);  // here i should be incremented, not start
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
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
12. Letter Combinations of a Phone Number

    String[] chars = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<String>();
        if(digits == null || digits.length() == 0) return result;
        backtracking(result, new StringBuilder(), digits, 0);
        return result;
    }

    public void backtracking(List<String> result, StringBuilder sb, String digits, int index) {
        if(index == digits.length()) {
            result.add(sb.toString());
            return;
        }

        String str = chars[digits.charAt(index) - '0'];
        for(char c : str.toCharArray()) {
            sb.append(c);
            backtracking(result, sb, digits, index + 1);
            sb.setLength(sb.length() - 1);
        }
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
13. 490. Rat in a  Maze
     //Bfs - shortest path
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        int[][] dirs={{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        Queue < int[] > queue = new LinkedList < > ();
        queue.add(start);
        visited[start[0]][start[1]] = true;
        while (!queue.isEmpty()) {
            int[] s = queue.remove();
            if (s[0] == destination[0] && s[1] == destination[1])
                return true;
            for (int[] dir: dirs) {
                int x = s[0] + dir[0];
                int y = s[1] + dir[1];
                while (x >= 0 && y >= 0 && x < maze.length && y < maze[0].length && maze[x][y] == 0) {
                    x += dir[0];
                    y += dir[1];
                }
                if (!visited[x - dir[0]][y - dir[1]]) {
                    queue.add(new int[] {x - dir[0], y - dir[1]});
                    visited[x - dir[0]][y - dir[1]] = true;
                }
            }
        }
        return false;
    }
----------------------------------------------------------------------------------

    boolean solveMaze(int maze[][]){ 
	
		int sol[][] = new int[N][N]; 
  
        if (solveMazeUtil(maze, 0, 0, sol) == false) { 
            System.out.print("Solution doesn't exist"); 
            return false; 
        } 
        printSolution(sol); 
        return true; 
    } 
  
    /* A recursive utility function to solve Maze problem */
    boolean solveMazeUtil(int maze[][], int x, int y, int sol[][]){ 
        // if (x, y is goal) return true 
        if (x == N - 1 && y == N - 1 && maze[x][y] == 1) { 
            sol[x][y] = 1; 
            return true; 
        } 

        // Check if maze[x][y] is valid 
        if (isSafe(maze, x, y) == true) { 
            // mark x, y as part of solution path 
            sol[x][y] = 1; 
  
            /* Move forward in x direction */
            if (solveMazeUtil(maze, x + 1, y, sol)) 
                return true; 
  
            /* If moving in x direction doesn't give solution then Move down in y direction */
            if (solveMazeUtil(maze, x, y + 1, sol)) 
                return true; 
  
            /* If none of the above movements works then BACKTRACK: unmark x, y as part of solution path */
            sol[x][y] = 0;
			
            return false; 
        } 
  
        return false; 
    }
	
    /* A utility function to check if x, y is valid index for N*N maze */
    boolean isSafe(int maze[][], int x, int y){ 
        // if (x, y outside maze) return false 
        return (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1); 
    } 

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
