/*

1. Sort Colors(Dutch National Flag alg)
2. Maximum Sum Subarray of Size K (easy)
3. Single Number II
4. Majority Element N/2
5. Majority Element N/3
6. 287. Find the Duplicate Number
7. 239. Sliding Window Maximum
8. Find subarray with given sum | Set 1
9. Find subarray with given sum | Set 2 (Handles Negative Numbers)
10. 15. 3Sum
11. 42. Trapping Rain Water
12. 11. Container With Most Water
13. Minimum Number of Platforms Required for a Railway/Bus Station 
14. Reverse an array in groups of given size
15. 941. Valid Mountain Array
16. 16. 56. Merge Intervals


*/
*-----------------------------------------------------------------------------------------------------------------------------------------------*
// 1. Sort Colors(Dutch National Flag alg)

    public void sortColors(int[] nums) {
        
      int low = 0,mid = 0, high;
      high = nums.length -1;
      while(mid<=high){  // impt <=
          if(nums[mid] == 0){
              swap(low,mid, nums);
              low++;
              mid++;
          }else if(nums[mid] == 1){
              mid++;
          }else{
              swap(mid,high, nums);
              high--;
          }  
      }
      
    }
    
    public void swap(int l, int r, int[] nums){
        int temp = nums[l];
        nums[l]=nums[r];
        nums[r]=temp;
    }


*-----------------------------------------------------------------------------------------------------------------------------------------------*
// 2. Maximum Sum Subarray of Size K (easy)

  public static int findMaxSumSubArray(int k, int[] arr) {
    int windowSum = 0, maxSum = 0;
    int windowStart = 0;
    for (int windowEnd = 0; windowEnd < arr.length; windowEnd++) {
      windowSum += arr[windowEnd]; // add the next element
      // slide the window, we don't need to slide if we've not hit the required window size of 'k'
      if (windowEnd >= k - 1) {
        maxSum = Math.max(maxSum, windowSum);
        windowSum -= arr[windowStart]; // subtract the element going out
        windowStart++; // slide the window ahead
      }
    }

    return maxSum;
  }
  
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/* 3. Single Number II
 
Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

*/

    public int singleNumber(int[] nums) {
        int result = 0; 
        int x = 0, sum = 0; 
          
        for(int i=0; i<32; i++){ 
            sum = 0; 
            x = (1 << i);    // left shift 1 to ith position
            for(int j=0; j<nums.length; j++) { 
                if((nums[j] & x) == x) 
                  sum++; 
            } 
            if ((sum % 3) != 0) 
              result |= x; 
        } 
        return result; 
        
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*

// 4. Majority Element N/2

	public static Integer getMajorityElement(int[] array) {

		if (array == null || array.length == 0) return null;
		
		Integer candidate = null;
		int count = 0;
		for (int i = 0; i < array.length; i++) {
			if (count == 0) {
				candidate = array[i];
				count = 1;
			} else {
				if (candidate == array[i]) {
					count++;
				} else {
					count--;
				}
			}
		}

		if (count == 0) {
			return null;
		}

		 
		count = 0;
		for (int i = 0; i < array.length; i++) {
			if (candidate == array[i]) {
				count++;
			}
		}
		return (count > array.length / 2) ? candidate : null;
	}
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*

// 5. Majority Element N/3

    public int repeatedNumber(final List<Integer> a) {
        
        int count1 = 0, count2 = 0;
        int first = Integer.MAX_VALUE, second = Integer.MAX_VALUE;
        int N = a.size();
		// Step 1
        for(int i=0; i<N; i++){
            int current = a.get(i);
            
            if(first == current) count1++;
            else if(second == current) count2++;
            else if(count1 == 0){
                count1++;
                first = current;
            }
            else if(count2 == 0){
                count2++;
                second = current;
            }else{
                count1--;
                count2--;
            }
            
        }
		//Step 2 check if the candidates are majority
        count1 = count2 = 0;
        for(int i=0; i<N; i++){
            int current = a.get(i);
            
            if(first == current) count1++;
            else if(second == current) count2++;           
        }
        
        if(count1>N/3) return first;
        
        if(count2>N/3) return second;      
        
        return -1;
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*6. 287. Find the Duplicate Number https://leetcode.com/articles/find-the-duplicate-number/

You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2). --> below is O(n) solution 
There is only one duplicate number in the array, but it could be repeated more than once

Floyd's Tortoise and Hare (Cycle Detection)
*/

  public int findDuplicate(int[] nums) {
    // Find the intersection point of the two runners.
    int tortoise = nums[0];
    int hare = nums[0];
    do {
      tortoise = nums[tortoise];
      hare = nums[nums[hare]];
    } while (tortoise != hare);

    // Find the "entrance" to the cycle.
    tortoise = nums[0];
    while (tortoise != hare) {
      tortoise = nums[tortoise];
      hare = nums[hare];
    }

    return hare;
  }

*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
7. 239. Sliding Window Maximum   https://leetcode.com/problems/sliding-window-maximum/discuss/65884/Java-O(n)-solution-using-deque-with-explanation

We scan the array from 0 to n-1, keep "promising" elements in the deque. The algorithm is amortized O(n) as each element is put and polled once.

At each i, we keep "promising" elements, which are potentially max number in window [i-(k-1),i] or any subsequent window. This means

 1. If an element in the deque and it is out of i-(k-1), we discard them. We just need to poll from the head, as we are using a deque and elements are ordered as the sequence in the array

 2. Now only those elements within [i-(k-1),i] are in the deque. We then discard elements smaller than a[i] from the tail. This is because if a[x] <a[i] and x<i, then a[x] has no chance to be the "max" in [i-(k-1),i], or any other subsequent window: a[i] would always be a better candidate.

 3. As a result elements in the deque are ordered in both sequence in array and their value. At each step the head of the deque is the max element in [i-(k-1),i]
*/

    public int[] maxSlidingWindow(int[] a, int k) {		
		if (a == null || k <= 0) {
			return new int[0];
		}
		int n = a.length;
		int[] r = new int[n-k+1];
		int ri = 0;
		// store index
		Deque<Integer> q = new ArrayDeque<>();
		for (int i = 0; i < a.length; i++) {
			// remove numbers out of range k
			while (!q.isEmpty() && q.peek() < i - k + 1) {
				q.poll();
			}
			// remove smaller numbers in k range as they are useless
			while (!q.isEmpty() && a[q.peekLast()] < a[i]) {
				q.pollLast();
			}
			// q contains index... r contains content
			q.offer(i);
			if (i >= k - 1) {
				r[ri++] = a[q.peek()];
			}
		}
		return r;
	}

*-----------------------------------------------------------------------------------------------------------------------------------------------*
// 8. Find subarray with given sum | Set 1

    int subArraySum(int arr[], int n, int sum){ 
        int curr_sum = arr[0], start = 0, i; 
  
        // Pick a starting point 
        for (i = 1; i <= n; i++) { 
            // If curr_sum exceeds the sum, 
            // then remove the starting elements 
            while (curr_sum > sum && start < i - 1) { 
                curr_sum = curr_sum - arr[start]; 
                start++; 
            } 
  
            // If curr_sum becomes equal to sum, then return true 
            if (curr_sum == sum) { 
                int p = i - 1; 
                System.out.println( 
                    "Sum found between indexes " + start 
                    + " and " + p); 
                return 1; 
            } 
  
            // Add this element to curr_sum 
            if (i < n) 
                curr_sum = curr_sum + arr[i]; 
        } 
  
        System.out.println("No subarray found"); 
        return 0; 
    } 

*-----------------------------------------------------------------------------------------------------------------------------------------------*
//9. Find subarray with given sum | Set 2 (Handles Negative Numbers)   ----> Not able to remember the logic at first glance 

    public static void subArraySum(int[] arr, int n, int sum) { 
        //cur_sum to keep track of cummulative sum till that point 
        int cur_sum = 0; 
        int start = 0; 
        int end = -1; 
        HashMap<Integer, Integer> hashMap = new HashMap<>(); 
  
        for (int i = 0; i < n; i++) { 
            cur_sum = cur_sum + arr[i]; 
            //check whether cur_sum - sum = 0, if 0 it means 
            //the sub array is starting from index 0- so stop 
            if (cur_sum - sum == 0) { 
                start = 0; 
                end = i; 
                break; 
            } 
            //if hashMap already has the value, means we already  
            // have subarray with the sum - so stop 
            if (hashMap.containsKey(cur_sum - sum)) { 
                start = hashMap.get(cur_sum - sum) + 1; 
                end = i; 
                break; 
            } 
            //if value is not present then add to hashmap 
            hashMap.put(cur_sum, i); 
  
        } 
        // if end is -1 : means we have reached end without the sum 
        if (end == -1) { 
            System.out.println("No subarray with given sum exists"); 
        } else { 
            System.out.println("Sum found between indexes " 
                            + start + " to " + end); 
        } 
  
    } 
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
10. 15. 3Sum

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

*/

    public List<List<Integer>> threeSum(int[] nums) {
        
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        
        Arrays.sort(nums);
        
        int n = nums.length;
        
        for(int i=0; i<n-2; i++){
            if (i == 0 || (i > 0 && nums[i] != nums[i-1])) {
                int start = i+1;
                int end = n-1;
                while(start<end){
                    int sum = nums[i] + nums[start] + nums[end];
                    if(sum==0){
                       result.add(Arrays.asList(nums[i], nums[start], nums[end]));
                        
                       int prev = nums[end];
                       while(end>=0 && prev == nums[end])
                         end--;
                    
                       prev = nums[start];
                       while(start<end && prev == nums[start])
                         start++;
                        
                    }else if (sum>0)
                        end--;
                    else
                        start++;   
                }
            }
            
        }
        
        return result;
        
    }

*-----------------------------------------------------------------------------------------------------------------------------------------------*
// 11. 42. Trapping Rain Water

	public int trap(int[] height) {
	  int n = height.length;
	  if (n <= 2) return 0;
	  // pre-compute
	  int[] leftMax = new int[n];
	  int[] rightMax = new int[n];
	  leftMax[0] = height[0]; // init
	  rightMax[n - 1] = height[n - 1];
	  for (int i = 1, j = n - 2; i < n; ++i, --j) {
		leftMax[i] = Math.max(leftMax[i - 1], height[i]);
		rightMax[j] = Math.max(rightMax[j + 1], height[j]);
	  }
	  // water
	  int totalWater = 0;
	  for (int k = 1; k < n - 1; ++k) { // do not consider the first and the last places
		int water = Math.min(leftMax[k - 1], rightMax[k + 1]) - height[k];
		totalWater += (water > 0) ? water : 0;
	  }
	  return totalWater;
	}

    ---------------------------------------------------
	
    public int trap(int[] height) {
        
        if(height == null || height.length==0 || height.length == 1 || height.length==2) return 0;
        
        int i = 0, j = height.length -1;
        int leftMax = 0, rightMax = 0, result = 0;
        
        while(i<=j){
            if(height[i]<height[j]){
                if(leftMax < height[i]){
                    leftMax = height[i];
                }else{
                    result += leftMax - height[i]; 
                }
                i++;
            }else{
                if(rightMax < height[j]){
                    rightMax = height[j];
                }else{
                    result += rightMax - height[j]; 
                }
                j--;      
            }
        }
        
        return result;
        
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
12. 11. Container With Most Water

*/

    public int maxArea(int[] height) {
        
        if(height == null || height.length == 0) return 0;
        
        int maxArea = 0;
        int left = 0, right = height.length-1;
        
        while(left < right){
            int min = Math.min(height[left], height[right]);
            maxArea = Math.max(maxArea, min * (right-left));
            if(height[left] < height[right]){
                left++;
            }else{
                right--;
            }
        }
        
        return maxArea;
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
13. Minimum Number of Platforms Required for a Railway/Bus Station  https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/
*/

    int findPlatform(int arr[], int dep[], int n) { 
        // Sort arrival and departure arrays 
        Arrays.sort(arr); 
        Arrays.sort(dep); 
  
        // plat_needed indicates number of platforms needed at a time 
        int plat_needed = 1, result = 1; 
        int i = 1, j = 0; 
  
        // Similar to merge in merge sort to process all events in sorted order 
        while (i < n && j < n) { 
  
            // If next event in sorted order is arrival, increment count of platforms needed 
            if (arr[i] <= dep[j]) { 
                plat_needed++; 
                i++; 
            } 
  
            // Else decrement count of platforms needed 
            else if (arr[i] > dep[j]) { 
                plat_needed--; 
                j++; 
            } 
  
            // Update result if needed 
            if (plat_needed > result) 
                result = plat_needed; 
        } 
  
        return result; 
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
14. Reverse an array in groups of given size

Given an array, reverse every sub-array formed by consecutive k elements.

*/

    void reverse(int arr[], int n, int k){ 
	
        for (int i = 0; i < n; i += k) { 
            int left = i; 
      
            // to handle case when k is not multiple of n
			
            int right = Math.min(i + k - 1, n - 1); 
            int temp; 
              
            // reverse the sub-array [left, right] 
            while (left < right) 
            { 
                temp=arr[left]; 
                arr[left]=arr[right]; 
                arr[right]=temp; 
                left+=1; 
                right-=1; 
            } 
        } 
    } 
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*
15. 941. Valid Mountain Array

Given an array A of integers, return true if and only if it is a valid mountain array.

Recall that A is a mountain array if and only if:


    1. A.length >= 3
	2. There exists some i with 0 < i < A.length - 1 such that:
		A[0] < A[1] < ... A[i-1] < A[i]
		A[i] > A[i+1] > ... > A[A.length - 1]

*/
    public boolean validMountainArray(int[] A) {
        
        int N = A.length;
        int i = 0;

        // walk up
        while (i+1 < N && A[i] < A[i+1])
            i++;

        // peak can't be first or last
        if (i == 0 || i == N-1)
            return false;

        // walk down
        while (i+1 < N && A[i] > A[i+1])
            i++;

        return i == N-1;
        
    }
	

*-----------------------------------------------------------------------------------------------------------------------------------------------*
/*

16. 56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:

Input: intervals = [[1,4],[2,3]]
Output: [[1,4]]

*/
    public int[][] merge(int[][] intervals) {
        
        if(intervals.length <= 1) return intervals;
        
        List<int[]> result = new ArrayList<int[]>();
        
        Arrays.sort(intervals, (nums1, nums2) -> nums1[0] - nums2[0]);
        
        for(int[] curInterval: intervals){
            
            if(result.size() == 0 || result.get(result.size()-1)[1] < curInterval[0]){
                result.add(curInterval);
            }else{
                result.get(result.size()-1)[1] = Math.max(result.get(result.size()-1)[1],curInterval[1]); // Using Max for coveing up Example 2 test case.
            }
            
        }
        
        return result.toArray(new int[result.size()][]);
    }
	
*-----------------------------------------------------------------------------------------------------------------------------------------------*