Trees :
1. Given a binary tree, find size of largest binary search subtree in this binary tree.(Largest BST in Binary Tree) Tushar Roy
2. Lowest Common Ancestor In A BST
3. 94. Binary Tree Inorder Traversal
4. 102. Binary Tree Level Order Traversal
5. Pre Order(not completed) and Post Order(not completed)
6. 297. Serialize and Deserialize Binary Tree
7. 173. Binary Search Tree Iterator
8. 938. Range Sum of BST
9. 669. Trim a Binary Search Tree
10. 1110. Delete Nodes And Return Forest
11. 662. Maximum Width of Binary Tree (not same level , litlle tricky)
------
12. 105. Construct Binary Tree from Preorder and Inorder Traversal
13. 106. Construct Binary Tree from Inorder and Postorder Traversal
14. 108. Convert Sorted Array to Binary Search Tree
------
15. 99. Recover Binary Search Tree
16. 783. Minimum Distance Between BST Nodes
    530. Minimum Absolute Difference in BST
17. 110. Balanced Binary Tree
18. 114. Flatten Binary Tree to Linked List
19. 437. Path Sum III
20. 863. All Nodes Distance K in Binary Tree
21. 222. Count Complete Tree Nodes
22. Find Leaves of Binary Tree (Java) LeetCode – Premium (https://www.programcreek.com/ )
23. Find distance between two nodes of a Binary Tree
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
Trie:
1. Longest Common Prefix

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

1. Given a binary tree, find size of largest binary search subtree in this binary tree.(Largest BST in Binary Tree) Tushar Roy

class MinMax{
    int min;
    int max;
    boolean isBST;
    int size ;
    
    MinMax(){
        min = Integer.MAX_VALUE;
        max = Integer.MIN_VALUE;
        isBST = true;
        size = 0;
    }
}


    private MinMax largest(Node root){
        //if root is null return min as Integer.MAX and max as Integer.MIN		
        if(root == null){
            return new MinMax();
        }
        
        //postorder traversal of tree. First visit left and right then
        //use information of left and right to calculate largest BST.
		
        MinMax leftMinMax = largest(root.left);
        MinMax rightMinMax = largest(root.right);
        
        MinMax m = new MinMax();
        
        //if either of left or right subtree says its not BST or the data
        //of this node is not greater/equal than max of left and less than min of right
        //then subtree with this node as root will not be BST. 
        //Return false and max size of left and right subtree to parent
		
        if(leftMinMax.isBST == false || rightMinMax.isBST == false || (leftMinMax.max > root.data || rightMinMax.min <= root.data)){
            m.isBST = false;
            m.size = Math.max(leftMinMax.size, rightMinMax.size);
			// not required 
			m.min = root.left != null ? leftMinMax.min : root.data;
			m.max = root.right != null ? rightMinMax.max : root.data;
			
            return m;
        }
        
        //if we reach this point means subtree with this node as root is BST.
        //Set isBST as true. Then set size as size of left + size of right + 1.
        //Set min and max to be returned to parent.
        m.isBST = true;
        m.size = leftMinMax.size + rightMinMax.size + 1;
     
        //if root.left is null then set root.data as min else
        //take min of left side as min
        m.min = root.left != null ? leftMinMax.min : root.data;
  
        //if root.right is null then set root.data as max else
        //take max of right side as max.
        m.max = root.right != null ? rightMinMax.max : root.data;
   
        return m;
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

2. Lowest Common Ancestor In A BST

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) { 
        
        if(root == null) return null;
        
        if(root.val < Math.min(q.val, p.val ))
            return lowestCommonAncestor(root.right, p, q);
		
        else if(root.val > Math.max(q.val, p.val ))
           return lowestCommonAncestor(root.left, p, q);
	   
        else
           return root;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
3. 94. Binary Tree Inorder Traversal

    public void helper(TreeNode root, List < Integer > res) {
        if (root != null) {
            if (root.left != null) {
                helper(root.left, res);
            }
            res.add(root.val);
            if (root.right != null) {
                helper(root.right, res);
            }
        }
    }
	
-----------------------------------------

    public List < Integer > inorderTraversal(TreeNode root) {
        List < Integer > res = new ArrayList < > ();
        Stack < TreeNode > stack = new Stack < > ();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            res.add(curr.val);
            curr = curr.right;
        }
        return res;
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
4. 102. Binary Tree Level Order Traversal

    public List<List<Integer>> levelOrder(TreeNode root) {
        
        if (root == null) 
            return new ArrayList<List<Integer>>();              

        List<List<Integer>> result = new  ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();

        queue.add(root);
        List<Integer> templist;
        
        while(!queue.isEmpty()){
            int size = queue.size();
            int i = 0;
            templist = new ArrayList<Integer>();
            
            while(i++<size){
                TreeNode node = queue.poll();
                templist.add(node.val);

                if(node.left != null){  
                    queue.add(node.left);
                }

                if(node.right != null){
                    queue.add(node.right); 
                }
            }
           result.add(templist);            
        }
        return result;     
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
6. 297. Serialize and Deserialize Binary Tree
// Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        
        if(root == null) return "";
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        StringBuilder str = new StringBuilder();
        que.add(root);
        str.append(String.valueOf(root.val) + " ");
        
        while(!que.isEmpty()){
            TreeNode node = que.poll();

            if(node.left!=null){
              que.add(node.left);
              str.append(String.valueOf(node.left.val)+ " ");
            }else{
              str.append("X" + " ");  
            }
            if(node.right!=null){
              que.add(node.right);
              str.append(String.valueOf(node.right.val)+ " ");
            }else{
              str.append("X" + " ");  
            }
        }
        
        return str.toString();    
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.length()==0) return null;
        
        String[] values = data.split(" ");
        
        Queue<TreeNode> que = new LinkedList<TreeNode>(); 

        int n = 1;
        TreeNode root =new TreeNode(Integer.valueOf(values[0]));
        que.add(root);
        
        while(!que.isEmpty() && n < values.length){
            
           TreeNode node = que.poll();
           if(!values[n].equals("X")){ 
               TreeNode left = new TreeNode(Integer.valueOf(values[n]));
               node.left = left;
               que.add(left);
           }else{
               node.left = null; 
           }
            
           n++;
           if(!values[n].equals("X")){ 
               TreeNode right = new TreeNode(Integer.valueOf(values[n]));
               node.right = right;
               que.add(right);
           }else{
              node.right = null; 
           }
           n++;
        }
        
        return root;
        
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
7. 173. Binary Search Tree Iterator


class BSTIterator {

    ArrayList<Integer> nodesSorted;
    int index;

    public BSTIterator(TreeNode root) {

        // Array containing all the nodes in the sorted order
        this.nodesSorted = new ArrayList<Integer>();
        
        // Pointer to the next smallest element in the BST
        this.index = -1;
        
        // Call to flatten the input binary search tree
        this._inorder(root);
    }

    private void _inorder(TreeNode root) {

        if (root == null) {
            return;
        }

        this._inorder(root.left);
        this.nodesSorted.add(root.val);
        this._inorder(root.right);
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return this.nodesSorted.get(++this.index);
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return this.index + 1 < this.nodesSorted.size();
    }
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
8. 938. Range Sum of BST


    public int rangeSumBST(TreeNode root, int L, int R) {
        
        if(root == null) return 0;
        if(root.val < L) return rangeSumBST(root.right, L, R);
        if(root.val > R) return rangeSumBST(root.left, L, R);
        return root.val + rangeSumBST(root.right, L, R) + rangeSumBST(root.left, L, R);
        
    }
	
	----------------------------------------
	
    public int rangeSumBST(TreeNode root, int L, int R) {
        int ans = 0;
        Stack<TreeNode> stack = new Stack();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node != null) {
                if (L <= node.val && node.val <= R)
                    ans += node.val;
                if (L < node.val)
                    stack.push(node.left);
                if (node.val < R)
                    stack.push(node.right);
            }
        }
        return ans;
    }


*--------------------------------------------------------*--------------------------------------------------------------------------------------*
9. 669. Trim a Binary Search Tree

    public TreeNode trimBST(TreeNode root, int L, int R) {
        if (root == null) return root;
        if (root.val > R) return trimBST(root.left, L, R);
        if (root.val < L) return trimBST(root.right, L, R);

        root.left = trimBST(root.left, L, R);
        root.right = trimBST(root.right, L, R);
        return root;
    }
	
	----------------------------------------
	
    public TreeNode trimBST(TreeNode root, int L, int R) {
        int l = L, r = R; 
        if(root == null) return root;
        
        while(root.val < l || root.val > r){
            if(root.val < l) root = root.right;
            if(root.val > r) root = root.left;
        }
        
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        boolean adjusted = false;
        
        while(!stack.isEmpty()){
            TreeNode node = stack.pop();

            if(node.left != null && node.left.val < l){
                node.left = node.left.right;
                adjusted = true;
            }
            if(node.right != null && node.right.val > r){
                node.right = node.right.left;
                adjusted = true;
            }
            if(!adjusted){
                if(node.left != null) stack.push(node.left);
                if(node.right != null) stack.push(node.right);
            }else{
                stack.push(node);
            }
            adjusted = false;
            
        }
        return root;
        
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
10. 1110. Delete Nodes And Return Forest

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> res = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        for (int i : to_delete) 
            set.add(i);

        if (!set.contains(root.val)) 
            res.add(root);
        
        dfs(root, set, res);
        return res;
    }

    private TreeNode dfs(TreeNode node, Set<Integer> set, List<TreeNode> res) {
        if (node == null) {
            return null;
        }
        node.left = dfs(node.left, set, res);
        node.right = dfs(node.right, set, res);
        if (set.contains(node.val)) {
            if (node.left != null) res.add(node.left);
            if (node.right != null) res.add(node.right);
            return null; // most imp thing 
        }
        return node;
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
11. 662. Maximum Width of Binary Tree (not same level , litlle tricky)

// DFS
  class Solution {
    int ans = 0;
    public int widthOfBinaryTree(TreeNode root) {
        List<Integer> levelLMN = new ArrayList<>(); 
        dfs(root, 1, 0, levelLMN);
        return ans;
    }

    private void dfs(TreeNode root, int id, int level, List<Integer> levelLMN) {
        if (root == null) return;
        if (level == levelLMN.size()) levelLMN.add(id);
        dfs(root.left , id * 2    , level + 1, levelLMN);
        dfs(root.right, id * 2 + 1, level + 1, levelLMN);
        ans = Math.max(id + 1 - levelLMN.get(level), ans);
    }
 }
-------------------------------------------------
  // BFS, need to add null nodes also, and after each level trim leftmost null and right most null, null node between first not null left and last not null are required foor ans
  class Solution {
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null)
            return 0;
        Deque<TreeNode> deque = new LinkedList<>(); // initialize deque
        deque.add(root); // add root node in deque
        int maxWidth = 0;
        while (!deque.isEmpty()) {
            int size = deque.size(); // size of current level
            maxWidth = Math.max(maxWidth, size);
            while (size-- > 0) {
                TreeNode node = deque.poll();
                if (node == null) { // node was null then to maintain add both left and right as null
                    deque.add(null);
                    deque.add(null);
                } else {
                    deque.add(node.left);
                    deque.add(node.right);
                }
            }
            while (!deque.isEmpty() && deque.peekFirst() == null)
                deque.pollFirst(); // remove all the null from the start until encounter first last non null node of level
            while (!deque.isEmpty() && deque.peekLast() == null)
                deque.pollLast(); // remove all the null from the last until encounter last non null node of level
        }
        return maxWidth;
    }
  }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
12. 105. Construct Binary Tree from Preorder and Inorder Traversal

class Solution {
    
    public int preIndex = 0;
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        
        return buildTreeUtil(preorder, inorder, 0, preorder.length-1, preorder.length);
        
    }
    
    public TreeNode buildTreeUtil(int[] preorder, int[] inorder, int left, int right, int size) {
        
        if(left > right || preIndex > size) return null;
        
        TreeNode node = new TreeNode(preorder[preIndex++]);
        
        int inorderIndex = searchIndex(inorder, left, right, node.val);
        
        node.left = buildTreeUtil(preorder, inorder, left, inorderIndex-1, size);
        node.right = buildTreeUtil(preorder, inorder, inorderIndex + 1 , right, size);
        
        return node;  
    }
    
    public int searchIndex(int[] inorder, int left, int right, int key){
        for(int i = left; i <= right; i++){
            if (inorder[i] == key) 
                return i; 
        }
        return -1;
    }

}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
13. 106. Construct Binary Tree from Inorder and Postorder Traversal

class Solution {
    
    public int index=0;
	
    public TreeNode buildTree(int[] inorder, int[] postorder) {
		
        if (inorder.length != postorder.length) return null;
        index = postorder.length-1;
        return buildTreeUtil(inorder, postorder, 0, inorder.length-1);
        
    }
   
    public TreeNode buildTreeUtil(int[] inorder, int[] postorder, int left, int right) {
        
        if(left>right || index<0) return null;
        
        TreeNode node = new TreeNode(postorder[index--]);
        
        int inorderIndex = search(node.val, inorder, left, right);
       
        node.right = buildTreeUtil(inorder, postorder, inorderIndex+1, right);
        node.left = buildTreeUtil(inorder, postorder, left, inorderIndex-1);
        
        return node;
    }
    
    public int search(int value, int[] inorder, int l, int r){
        for(int i=l; i<=r; i++){
            if(inorder[i] == value)
                return i;
        }
        return -1;
    }
    
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
14. 108. Convert Sorted Array to Binary Search Tree

    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums == null || nums.length==0) return null;
        return sortedArrayToBST(nums, 0, nums.length-1);
    }
    
    private TreeNode sortedArrayToBST(int[] A, int left, int right){
        if(left>right) return null;
        if (left==right) return new TreeNode(A[left]);
        int mid = (left + right)/2;
        TreeNode root = new TreeNode(A[mid]);
        root.left = sortedArrayToBST(A, left, mid-1);
        root.right = sortedArrayToBST(A, mid+1, right);
        return root;
    }
  }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

15. 99. Recover Binary Search Tree https://leetcode.com/problems/recover-binary-search-tree/discuss/32535/No-Fancy-Algorithm-just-Simple-and-Powerful-In-Order-Traversal

    TreeNode firstElement, secondElement, prevElement  = null;
    
    public void recoverTree(TreeNode root) {
        traverse(root);
        
        // Swap the values of the two nodes
        int temp = firstElement.val;
        firstElement.val = secondElement.val;
        secondElement.val = temp;
    }
    
    private void traverse(TreeNode root) {
        
        if (root == null)
            return;
            
        traverse(root.left);
        
        if(prevElement!=null){
            
            if (firstElement == null && prevElement.val >= root.val) 
                firstElement = prevElement;
            
            if (firstElement != null && prevElement.val >= root.val) 
                secondElement = root;
            
        }
        prevElement = root;

        traverse(root.right);
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

16. 783. Minimum Distance Between BST Nodes
    530. Minimum Absolute Difference in BST
	
    Integer res = Integer.MAX_VALUE, pre = null;
    public int minDiffInBST(TreeNode root) {
        if (root.left != null) minDiffInBST(root.left);
        if (pre != null) res = Math.min(res, root.val - pre);
        pre = root.val; // after processing left, root --> right node will be set to pre(as right is always greater node in InOrderTraversal)
        if (root.right != null) minDiffInBST(root.right);
        return res;
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

17. 110. Balanced Binary Tree

    // botton up approach - O(n)
    public boolean isBalanced(TreeNode root) {
        return getHeight(root) != -1;
    }
    
    private int getHeight(TreeNode node) {
        if (node == null) return 0;

        int left = getHeight(node.left);
        int right = getHeight(node.right);

        // left, right subtree is unbalanced or cur tree is unbalanced
        if (left == -1 || right == -1 || Math.abs(left - right) > 1) return -1;

        return Math.max(left, right) + 1;
    }
	
	------------------------------------
	// based on height - greater than O(n*n) 
	private boolean result = true;
	public boolean isBalanced(TreeNode root) {
		maxDepth(root);
		return result;
	}

	public int maxDepth(TreeNode root) {
		if (root == null)
			return 0;
		int l = maxDepth(root.left);
		int r = maxDepth(root.right);
		if (Math.abs(l - r) > 1)
			result = false;
		return 1 + Math.max(l, r);
	}
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

18. 114. Flatten Binary Tree to Linked List
    // depends on problem to use stack or queue
    public void flatten(TreeNode root) {
        
        if (root == null) return;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        stk.push(root);
        
        while (!stk.isEmpty()){
            
            TreeNode curr = stk.pop();
            
            if (curr.right!=null)  
                 stk.push(curr.right);
            
            if (curr.left!=null)  
                 stk.push(curr.left);
            
            if (!stk.isEmpty()) 
                 curr.right = stk.peek();
            
            curr.left = null; 
    
        }
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
19. 437. Path Sum III

    public int pathSum(TreeNode root, int sum) {
        if(root==null) return 0;
        return findPathSum(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    
    public int findPathSum(TreeNode root, int sum) {
        if(root==null) return 0;
        
        return (root.val == sum ? 1 : 0) + 
               findPathSum(root.left, sum - root.val) + 
               findPathSum(root.right, sum - root.val);
          
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
20. 863. All Nodes Distance K in Binary Tree


    void printkdistanceNodeDown(Node node, int k){ 
        // Base Case 
        if (node == null || k < 0) 
            return; 
   
        // If we reach a k distant node, print it 
        if (k == 0){ 
            System.out.print(node.data); 
            System.out.println(""); 
            return; 
        } 

        // Recur for left and right subtrees 
        printkdistanceNodeDown(node.left, k - 1); 
        printkdistanceNodeDown(node.right, k - 1); 
    } 
   
    // Prints all nodes at distance k from a given target node. The k distant nodes may be upward or downward.This function 
    // Returns distance of root from target node, it returns -1, if target node is not present in tree rooted with root. 
    int printkdistanceNode(Node node, Node target, int k){ 
        // Base Case 1: If tree is empty, return -1 
        if (node == null) 
            return -1; 
   
        // If target is same as root.  Use the downward function to print all nodes at distance k in subtree rooted with target or root 
        if (node == target){ 
            printkdistanceNodeDown(node, k); 
            return 0; 
        } 
   
        // Recur for left subtree 
        int dl = printkdistanceNode(node.left, target, k); 
   
        // Check if target node was found in left subtree 
        if (dl != -1){ 
            // If root is at distance k from target, print root , Note that dl is Distance of root's left child from target   
            if (dl + 1 == k) { 
                System.out.print(node.data); 
                System.out.println(""); 
            }    
            // Else go to right subtree and print all k-dl-2 distant nodes, Note that the right child is 2 edges away from left child 
            else
                printkdistanceNodeDown(node.right, k - dl - 2); 
   
            // Add 1 to the distance and return value for parent calls 
            return 1 + dl; 
        } 
 
        int dr = printkdistanceNode(node.right, target, k); 
        if (dr != -1)  
        { 
            if (dr + 1 == k)  
            { 
                System.out.print(node.data); 
                System.out.println(""); 
            }  
            else 
                printkdistanceNodeDown(node.left, k - dr - 2); 
            return 1 + dr; 
        } 
   
        // If target was neither present in left nor in right subtree 
        return -1; 
    } 
	
	-------------------------------
  class Solution {
    List<Integer> result = new ArrayList<Integer>();
    
    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
       distanceKUtil(root, target, K); 
       return result;
    }
    
    public int distanceKUtil(TreeNode root, TreeNode target, int K) {
        
        if(root == null) return -1;
        
        if(root == target){
            printKNodesDown(root, K);
            return 1;
        }
            
        int leftSubTree = distanceKUtil(root.left, target, K);
        if(leftSubTree != -1){
            
            if(leftSubTree == K)
                result.add(root.val);
            else
                printKNodesDown(root.right, K-leftSubTree-1);
            
            return leftSubTree+1;
        }
        
        int rightSubTree = distanceKUtil(root.right, target, K);
        if(rightSubTree != -1){
            
            if(rightSubTree == K)
                result.add(root.val);
            else
                printKNodesDown(root.right, K-rightSubTree-1);
            
            return rightSubTree+1;
        }
        
        return -1;
    }
    
    public void printKNodesDown(TreeNode root, int K){
        if(K<0 || root == null) return;
        if(K==0){
           result.add(root.val);
           return;
        }
        
        printKNodesDown(root.left, K-1);
        printKNodesDown(root.right, K-1);
    }
    
 } 
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*	

21. 222. Count Complete Tree Nodes

    // complexity is O(N)
    public int countNodes(TreeNode root) {
        
        if(root == null) return 0;
        
        int left = countNodes(root.left);
        int right = countNodes(root.right);
        
        return 1 + left + right;
        
    }
-----------------------------------------------------------------------

    // Optimal solution - O(h) where h is height of tree.
    public int countNodes(TreeNode root) {
        if(root==null) return 0;
        
        int lHeight = 0, rHeight = 0;
        
        TreeNode current = root;
        // calculate left height
        while(current!=null){
          current = current.left;
          lHeight++;
        }
        // calculate right height
        current = root;
        while(current!=null){
          current = current.right;
          rHeight++;
        }
        // if both heights are same then the complete binary tree is perfect binary tree, else do the same for remaining
        if(lHeight == rHeight)
            return (int) Math.pow(2, rHeight) - 1;
        
        return 1 + countNodes(root.left) + countNodes(root.right);
        
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

22. Find Leaves of Binary Tree (Java) LeetCode – Premium (https://www.programcreek.com/ )

https://www.programcreek.com/2014/07/leetcode-find-leaves-of-binary-tree-java/#:~:text=Given%20a%20binary%20tree%2C%20collect,until%20the%20tree%20is%20empty.&text=Returns%20%5B4%2C%205%2C%203,2%5D%2C%20%5B1%5D.

public List<List<Integer>> findLeaves(TreeNode root) {
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    helper(result, root);
    return result;
}
 
// traverse the tree bottom-up recursively
private int helper(List<List<Integer>> list, TreeNode root){
    if(root==null)
        return -1;
 
    int left = helper(list, root.left);
    int right = helper(list, root.right);
    int curr = Math.max(left, right)+1;
 
    // the first time this code is reached is when curr==0,
    //since the tree is bottom-up processed.
    if(list.size()<=curr){
        list.add(new ArrayList<Integer>());
    }
 
    list.get(curr).add(root.val);
 
    return curr;
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

23. Find distance between two nodes of a Binary Tree

Ref: https://www.geeksforgeeks.org/find-distance-between-two-nodes-of-a-binary-tree/

We first find LCA of two nodes. Then we find distance from LCA to two nodes.

    public static Node LCA(Node root, int n1, int n2){ 
        if (root == null) return root; 
        if (root.value == n1 || root.value == n2) return root; 
  
        Node left = LCA(root.left, n1, n2); 
        Node right = LCA(root.right, n1, n2); 
  
        if (left != null && right != null) return root; 
        if (left != null) 
            return LCA(root.left, n1, n2); 
        else
            return LCA(root.right, n1, n2); 
    } 
	
	// Returns level of key k if it is present in 
    // tree, otherwise returns -1 
    public static int findLevel(Node root, int a, int level){ 
        if (root == null) return -1; 
        if (root.value == a) return level;
        int left = findLevel(root.left, a, level + 1); 
        if (left == -1) return findLevel(root.right, a, level + 1); 
        return left; 
    } 
  
    public static int findDistance(Node root, int a, int b){ 
        Node lca = LCA(root, a, b); 
        int d1 = findLevel(lca, a, 0); 
        int d2 = findLevel(lca, b, 0); 
        return d1 + d2; 
    } 

*--------------------------------------------------------*--------------------------------------------------------------------------------------*