
class TrieNode {

    // R links to node children
    private TrieNode[] links;

    private final int R = 26;

    private boolean isEnd;

    public TrieNode() {
        links = new TrieNode[R];
    }

    public boolean containsKey(char ch) {
        return links[ch -'a'] != null;
    }
    public TrieNode get(char ch) {
        return links[ch -'a'];
    }
    public void put(char ch, TrieNode node) {
        links[ch -'a'] = node;
    }
    public void setEnd() {
        isEnd = true;
    }
    public boolean isEnd() {
        return isEnd;
    }
}

class Trie {
    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char currentChar = word.charAt(i);
            if (!node.containsKey(currentChar)) {
                node.put(currentChar, new TrieNode());
            }
            node = node.get(currentChar);
        }
        node.setEnd();
    }
	
    // search a prefix or whole key in trie and
    // returns the node where search ends
    private TrieNode searchPrefix(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
           char curLetter = word.charAt(i);
           if (node.containsKey(curLetter)) {
               node = node.get(curLetter);
           } else {
               return null;
           }
        }
        return node;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
       TrieNode node = searchPrefix(word);
       return node != null && node.isEnd();
    }
	
}

----------------------------------------------------------------------------------------------------------------------------------------------
//HashMap

class TrieNode {
    private HashMap<Character, TrieNode> children = new HashMap<>();
    public boolean isEnd = false; 

    public void putChildIfAbsent(char ch) {
        children.putIfAbsent(ch, new TrieNode());
    }

    public TrieNode getChild(char ch) {
        return children.get(ch);
    }
}

class Trie {
     TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
       root = new TrieNode(); 
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode curr = root;
        for (char ch : word.toCharArray()) {
            curr.putChildIfAbsent(ch);
            curr = curr.getChild(ch);
        }
        curr.isEnd = true;
        
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode curr = root;
        for (char ch : word.toCharArray()) {
            curr = curr.getChild(ch);
            if (curr == null) {
                return false;
            }
        }
        return curr.isEnd;        
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode curr = root;
        for (char ch : prefix.toCharArray()) {
            curr = curr.getChild(ch);
            if (curr == null) {
                return false;
            }
        }
        return true;        
    }
}

----------------------------------------------------------------------------------------------------------------------------------------------
// Auto Complete using Trie

    public List<String> getWordsForPrefix(String pre){
    	List<String> results = new ArrayList<String>();
    	TrieNode node = root;
    	for(char c: pre.toCharArray()){
    		if(node.containsKey(c))
    		   node = node.get(c);
    		else
    		   return results;
    	}
    	
    	findAllWords(pre, node, results);
    	return results;
    }


    private void findAllWords(String prefix, TrieNode n, List<String> result){
        
    	if(n.isEnd()) result.add(prefix);
    	
    	for(char c='a'; c<='z'; c++){ 
    		TrieNode node = n.get(c);
    		if(node!=null)
    		  findAllWords(prefix + 'c', node, result);
    	}
    }

----------------------------------------------------------------------------------------------------------------------------------------------
**A Binary Heap is a Complete Binary Tree.

abstract class Heap {

    protected int capacity;
    protected int size;
    protected int[] items;

    public Heap() {
        this.capacity = 10;
        this.size = 0;
        this.items = new int[capacity];
    }
    
    public int getLeftChildIndex(int parentIndex) { return 2 * parentIndex + 1; }
    public int getRightChildIndex(int parentIndex) { return 2 * parentIndex + 2; }
    public int getParentIndex(int childIndex) { return (childIndex - 1) / 2; }
    
    public boolean hasLeftChild(int index) { return getLeftChildIndex(index) < size; }
    public boolean hasRightChild(int index) { return getRightChildIndex(index) < size; }
    public boolean hasParent(int index) { return getParentIndex(index) >= 0; }
    

    public int leftChild(int index) { return items[getLeftChildIndex(index)]; }
    public int rightChild(int index) { return items[getRightChildIndex(index)]; }
    public int parent(int index) { return items[getParentIndex(index)]; }
    
    public void swap(int indexOne, int indexTwo) {
        int temp = items[indexOne];
        items[indexOne] = items[indexTwo];
        items[indexTwo] = temp;
    }
    
    public void ensureCapacity() {
        if(size == capacity) {
            capacity = capacity << 1;
            items = Arrays.copyOf(items, capacity);
        }
    }
    
    public int peek() {
        isEmpty("peek");
        return items[0];
    }
    
    public void isEmpty(String methodName) {
        if(size == 0) {
            throw new IllegalStateException(
                "You cannot perform '" + methodName + "' on an empty Heap."
            );
        }
    }
    
    public int poll() {
        // Throws an exception if empty.
        isEmpty("poll");
        // Else, not empty
        int item = items[0];
        items[0] = items[size - 1];
        size--;
        heapifyDown();
        return item;
    }
    
    public void add(int item) {
        // Resize underlying array if it's not large enough for insertion
        ensureCapacity();
        // Insert value at the next open location in heap
        items[size] = item;
        size++;
        // Correct order property
        heapifyUp();
    }
    
    /** Swap values down the Heap. **/
    public abstract void heapifyDown();
    
    /** Swap values up the Heap. **/
    public abstract void heapifyUp();
}


class MaxHeap extends Heap {
    
    public void heapifyDown() {
        int index = 0;
        while(hasLeftChild(index)) {
            int smallerChildIndex = getLeftChildIndex(index);
            
            if(hasRightChild(index) && rightChild(index) > leftChild(index)) {
                smallerChildIndex = getRightChildIndex(index);
            }
            
            if(items[index] > items[smallerChildIndex]) {
                break;
            }
            else {
                swap(index, smallerChildIndex);
            }
            index = smallerChildIndex;
        }
    }
    
    public void heapifyUp() {
        int index = size - 1;
        
        while(hasParent(index) && parent(index) < items[index]) {
            swap(getParentIndex(index), index);
            index = getParentIndex(index);
        }
    }
}

class MinHeap extends Heap {
    
    public void heapifyDown() {
        int index = 0;
        while(hasLeftChild(index)) {
            int smallerChildIndex = getLeftChildIndex(index);
            
            if(hasRightChild(index) && rightChild(index) < leftChild(index)) {
                smallerChildIndex = getRightChildIndex(index);
            }
            
            if(items[index] < items[smallerChildIndex]) {
                break;
            }
            else {
                swap(index, smallerChildIndex);
            }
            index = smallerChildIndex;
        }
    }
    
    public void heapifyUp() {
        int index = size - 1;
        
        while(hasParent(index) && parent(index) > items[index]) {
            swap(getParentIndex(index), index);
            index = getParentIndex(index);
        }
    }
}
--------------------------------------------------------------------
//Merge K Sorted Arrays

  public static List<Integer> mergeSortedArrays(List<List<Integer>> arrays) {
        ArrayList<Integer> answer = new ArrayList<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> a[2] - b[2]);
		
        for(int i=0;i<arrays.size();i++)
            if(arrays.get(i) != null) 
              pq.add(new int[]{ i, 0, arrays.get(i).get(0) });
        
        while(!pq.isEmpty()) {
            int[] temp = pq.poll();
            answer.add(temp[2]);
            //Check if last element
            if (temp[1] >= arrays.get(temp[0]).size() - 1) continue;
            pq.add(new int[] { temp[0], temp[1] + 1, arrays.get(temp[0]).get(temp[1] + 1) });
        }

        return answer;
  }
  
--------------------------------------------------------------------