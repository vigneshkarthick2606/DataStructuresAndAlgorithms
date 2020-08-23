
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