1. 743. Network Delay Time
2. 787. Cheapest Flights Within K Stops
3. 886. Possible Bipartition
4. 79. Word Search    
5. 127. Word Ladder
6. Topological Sorting
   210. Course Schedule II 
7. Dijkstra’s shortest path algorithm in Java using PriorityQueue

8. Disjoint Set (Or Union-Find) 
   547. Friend Circles (Union-Find path compression) https://leetcode.com/problems/friend-circles/discuss/101336/Java-solution-Union-Find


*--------------------------------------------------------*--------------------------------------------------------------------------------------*
Other Graph Problems:

https://leetcode.com/problems/all-paths-from-source-to-target/discuss/118713/Java-DFS-Solution
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
1. Island Perimeter
2. Number of Distinct Islands
3. Max Area of Island 
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
1. 743. Network Delay Time

    public int networkDelayTime(int[][] times, int N, int K) {
        
        Map<Integer, List<int[]>> map = new HashMap<>();
        for(int[] time : times){
            map.putIfAbsent(time[0], new ArrayList<>());
            map.get(time[0]).add(new int[]{time[1], time[2]});
        }
        
        //distance, node into pq
        Queue<int[]> pq = new PriorityQueue<>((a,b) -> (a[0] - b[0]));
        
        pq.add(new int[]{0, K});
        
        Set<Integer> visited = new HashSet<>();
        int res = 0, nodes = N;
        
        while(!pq.isEmpty()){
            int[] cur = pq.remove();
            int curNode = cur[1];
            int curDist = cur[0];
			
            if(!visited.add(curNode)) continue;
            if(visited.size()==N) return curDist;
			
            res = curDist;
            if(map.containsKey(curNode)){
                for(int[] next : map.get(curNode)){
                    pq.add(new int[]{curDist + next[1], next[0]});
                }
            }
        }
        return visited.size()==N? res : -1;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

2. 787. Cheapest Flights Within K Stops
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        
        Map<Integer,List<int[]>> map=new HashMap<>();
		
        for(int[] f:flights){
            map.putIfAbsent(f[0],new ArrayList<>());
            map.get(f[0]).add(new int[]{f[1],f[2]});
        }
		
     /*   PriorityQueue<int[]> q=new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0],o2[0]);
            }
        }); */
		
        PriorityQueue<int[]> q = new PriorityQueue<>((a,b) -> a[0] - b[0]);
        
        q.offer(new int[]{0,src,K});
        
        while(!q.isEmpty())
        {
            int[] c=q.poll();
            int cost=c[0];
            int curr=c[1];
            int stop=c[2];
			
            if(curr==dst)
                return cost;
            
            if(stop>=0)
            {
                if(!map.containsKey(curr))
                    continue;
                
                for(int[] next:map.get(curr))
                {
                    q.add(new int[]{cost+next[1],next[0],stop-1});
                }
            }
        }
        
        return -1;
        
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

3. 886. Possible Bipartition

    public boolean possibleBipartition(int N, int[][] dislikes) {
       
        //Convert the Given into a undirected graph
        List<Integer>[] graph = new ArrayList[N];
        for (int i = 0; i < N; i++) {
          graph[i] = new ArrayList<>();
        }

        //construct the graph 
        for (int[] dislike : dislikes) {
          int u = dislike[0] - 1;
          int v = dislike[1] - 1;

          graph[u].add(v);
          graph[v].add(u);
        }

        //colors array - paint adjacent nodes with diffrent color
        //if two adjacent nodes have same color return false
        int[] colors = new int[N];

        for (int i = 0; i < N; i++) {
            
          // skip nodes which are already colored
          if (colors[i] != 0) {
            continue;
          }
          
          //dont forget to set curr source node color!
          colors[i] = 1;

          Queue<Integer> queue = new LinkedList<>();
          queue.add(i);

          while (!queue.isEmpty()) {
            int node = queue.poll();

            for (int adj : graph[node]) {
              if (colors[adj] == colors[node]) {
                return false;
              }

              if (colors[adj] == 0) {
                colors[adj] = -colors[node];
                queue.add(adj);
              }
            }
          }
        }

        return true;
        
    }
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
4. 79. Word Search    https://leetcode.com/problems/word-search/

    boolean[][] visited;
	
    public boolean exist(char[][] board, String word) {
        
        int row = board.length;
        if(row==0) return false;
        int col = board[0].length;
        
        visited = new boolean[row][col];
        
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                if(board[i][j] == word.charAt(0) && wordSearch(board, word, 0, i, j))
                    return true;
            }
        }
        
        return false;
        
        
    }
    
    public boolean wordSearch(char[][] grid, String word, int index, int i, int j){
        
        if(index==word.length()) return true;
        
        if(i<0 || i>=grid.length || j<0 || j>=grid[0].length || visited[i][j] == true ||
           grid[i][j] != word.charAt(index))
            return false;
        
        visited[i][j] = true;
        if(wordSearch(grid, word, index+1, i+1, j)||
           wordSearch(grid, word, index+1, i,   j+1) ||
           wordSearch(grid, word, index+1, i,   j-1) ||
           wordSearch(grid, word, index+1, i-1, j) )
            return true;
        
        visited[i][j] = false;
        
        return false;

    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
5. 127. Word Ladder

Input:

beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        
        HashSet<String> set = new HashSet<>(wordList);
        int level = 0;
        Queue<String> queue = new LinkedList<>();
        queue.add(beginWord);
        
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i=0; i<size; i++){
                String current = queue.poll();

                if(current.equals(endWord)) return level+1;
                
                for(int j=0; j<current.length(); j++){
                    char[] word = current.toCharArray();

                    for(char c='a'; c<='z'; c++){
                        word[j] = c;
                        String newString = new String(word);
                        
                        if(!newString.equals(current) && set.contains(newString)){
                           queue.add(newString);
                           set.remove(newString);
                        }
                    }

                }
                
            }
            
            level++;
        }
        
        return 0;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
6. Topological Sorting 

    
    // A recursive function used by topologicalSort  
    void topologicalSortUtil(int v, boolean visited[], Stack<Integer> stack){
		
        // Mark the current node as visited.  
        visited[v] = true;  
        Integer i;  
		
        // Recur for all the vertices adjacent to thisvertex  
        Iterator<Integer> it = adj.get(v).iterator();  
        while (it.hasNext()){  
            i = it.next();  
            if (!visited[i])  
                topologicalSortUtil(i, visited, stack);  
        }  
    
        // Push current vertex to stack which stores result  
        stack.push(new Integer(v));  
    }  
    
    // The function to do Topological Sort. It uses recursive topologicalSortUtil()  
    void topologicalSort(){  
        Stack<Integer> stack = new Stack<Integer>();  
    
        // Mark all the vertices as not visited  
        boolean visited[] = new boolean[V];  
        for (int i = 0; i < V; i++)  
            visited[i] = false;  
    
        // Call the recursive helper function to store Topological Sort starting from all vertices one by one  
        for (int i = 0; i < V; i++)  
            if (visited[i] == false)  
                topologicalSortUtil(i, visited, stack);  
    
        // Print contents of stack  
        while (stack.empty()==false)  
            System.out.print(stack.pop() + " ");  
    }  

--------------------------------------------------------------------
    210. Course Schedule II  (Topological Sorting - using Indegree (Kahn’s algorithm ))

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        
        List<Integer>[] graph = new ArrayList[numCourses];
        Queue<Integer> queue = new LinkedList<Integer>();
        int[] indegree = new int[numCourses];
        int[] answer = new int[numCourses];
        int index = 0;
        
        for(int i=0; i<numCourses; i++)
            graph[i] = new ArrayList<Integer>();
        
        for(int[] a: prerequisites){
            graph[a[1]].add(a[0]);
            indegree[a[0]]++;
        }
        
        for(int i=0; i<numCourses; i++){
            if(indegree[i]==0){
               queue.add(i);
               answer[index++] = i;
            }
                
        }
        
        while(!queue.isEmpty()){
            int node = queue.poll();
            for(int currNode: graph[node]){
                indegree[currNode]--;
                if(indegree[currNode] == 0){
                  queue.add(currNode);
                  answer[index++] = currNode;
                }
            }
        }
         
        // base case - topologicalSort not possible		 
        for(int a: indegree) 
            if(a>0)
                return new int[0];
        
        return answer;
        
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
7. Dijkstra’s shortest path algorithm in Java using PriorityQueue

public class DPQ { 
    private int dist[]; 
    private Set<Integer> settled; 
    private PriorityQueue<Node> pq; 
    private int V; // Number of vertices 
    List<List<Node>> adj;

    // Function for Dijkstra's Algorithm 
    public void dijkstra(List<List<Node> > adj, int src){ 
        this.adj = adj; 
  
        for (int i = 0; i < V; i++) 
            dist[i] = Integer.MAX_VALUE; 
  
        // Add source node to the priority queue 
        pq.add(new Node(src, 0)); 
  
        // Distance to the source is 0 
        dist[src] = 0; 
        while (settled.size() != V) { 
  
            // remove the minimum distance node from the priority queue  
            int u = pq.remove().node; 
  
            // adding the node whose distance is finalized 
            settled.add(u); 
  
            e_Neighbours(u); 
        } 
    } 

    // Function to process all the neighbours of the passed node 
    private void e_Neighbours(int u){ 
        int edgeDistance = -1; 
        int newDistance = -1; 
  
        // All the neighbors of v 
        for (int i = 0; i < adj.get(u).size(); i++) { 
            Node v = adj.get(u).get(i); 
  
            // If current node hasn't already been processed 
            if (!settled.contains(v.node)) { 
                edgeDistance = v.cost; 
                newDistance = dist[u] + edgeDistance; 
  
                // If new distance is cheaper in cost 
                if (newDistance < dist[v.node]) 
                    dist[v.node] = newDistance; 
  
                // Add the current node to the queue 
                pq.add(new Node(v.node, dist[v.node])); 
            } 
        } 
    } 
}

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
8. Disjoint Set (Or Union-Find) | Set 1 (Detect Cycle in an Undirected Graph)

    int find(int parent[], int i){ 
        if (parent[i] == -1) 
            return i; 
        return find(parent, parent[i]); 
    } 
  
    // A utility function to do union of two subsets 
    void Union(int parent[], int x, int y){ 
        int xset = find(parent, x); 
        int yset = find(parent, y); 
        parent[xset] = yset; 
    } 

    int isCycle( Graph graph){ 
        // Allocate memory for creating V subsets 
        int parent[] = new int[graph.V]; 
  
        // Initialize all subsets as single element sets 
        for (int i=0; i<graph.V; ++i) 
            parent[i]=-1; 
  
        // Iterate through all edges of graph, find subset of both vertices of every edge, if both subsets are same, then there is cycle in graph. 
        for (int i = 0; i < graph.E; ++i){ 
            int x = graph.find(parent, graph.edge[i].src); 
            int y = graph.find(parent, graph.edge[i].dest); 
  
            if (x == y) 
                return 1; 
  
            graph.Union(parent, x, y); 
        } 
        return 0; 
    } 
	
	Note that the implementation of union() and find() is naive and takes O(n) time in worst case. These methods can be improved to O(Logn) using Union by Rank or Height. 
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

class Graph { 
	int V, E; 
	Edge[] edge; 
  
	Graph(int nV, int nE){ 
		V = nV; 
		E = nE; 
		edge = new Edge[E]; 
		for (int i = 0; i < E; i++)  
		{ 
			edge[i] = new Edge(); 
		} 
	} 
	  
	// class to represent edge  
	class Edge{ 
		int src, dest; 
	} 
	  
	// class to represent Subset 
	class subset{ 
		int parent; 
		int rank; 
	} 
	
	// A utility function to find  set of an element i (uses path compression technique) 
	int find(subset [] subsets , int i){ 
	
		if (subsets[i].parent != i) 
			subsets[i].parent = find(subsets, subsets[i].parent); 
			return subsets[i].parent; 
	} 
	  
	// A function that does union of two sets of x and y (uses union by rank) 
	void Union(subset [] subsets, int x , int y ){ 
		int xroot = find(subsets, x); 
	    int yroot = find(subsets, y); 
	  
		if (subsets[xroot].rank < subsets[yroot].rank) 
			subsets[xroot].parent = yroot; 
		else if (subsets[yroot].rank < subsets[xroot].rank) 
			subsets[yroot].parent = xroot; 
		else{ 
			subsets[xroot].parent = yroot; 
			subsets[yroot].rank++; 
		} 
	} 	
	
	int isCycle(Graph graph){ 
		int V = graph.V; 
		int E = graph.E; 
	  
		subset [] subsets = new subset[V]; 
		
		for (int v = 0; v < V; v++) { 
			subsets[v] = new subset(); 
			subsets[v].parent = v; 
			subsets[v].rank = 0; 
		} 
	  
		for (int e = 0; e < E; e++) { 
			int x = find(subsets, graph.edge[e].src); 
			int y = find(subsets, graph.edge[e].dest); 
			if(x == y) 
				return 1; 
			Union(subsets, x, y); 
		} 
	return 0; 
	} 
	
}	

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

    class UnionFind {
        private int count = 0;
        private int[] parent, rank;
        
        public UnionFind(int n) {
            count = n;
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        public int find(int p) {
        	while (p != parent[p]) {
                parent[p] = parent[parent[p]];    // path compression by halving
                p = parent[p];
            }
            return p;
        }
        
        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) return;
            
            if (rank[rootQ] > rank[rootP]) {
                parent[rootP] = rootQ;
            }
            else if (rank[rootQ] < rank[rootP]){
                parent[rootQ] = rootP;
            }else{
                parent[rootP] = rootQ;
                rank[rootQ]++;
            }
            
            count--;
        }
        
        public int count() {
            return count;
        }
    }	
	
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
  