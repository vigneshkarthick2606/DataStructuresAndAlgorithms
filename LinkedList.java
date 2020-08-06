1. Reverse LinkedList 

    /* Recursive - reverse the linked list */
    Node reverse(Node head) { 
        if(head == null) { 
            return head; 
        } 
  
        // last node or only one node 
        if(head.next == null) { 
            return head; 
        } 
  
        Node newHeadNode = reverse(head.next); 
  
        // change references for middle chain 
        head.next.next = head; 
        head.next = null; 
  
        // send back new head node in every recursion 
        return newHeadNode; 
    }
	
-------------------------------------------------------------------
   
    /* Iterative - reverse the linked list */
    Node reverse(Node node) 
    { 
        Node prev = null; 
        Node current = node; 
        Node next = null; 
        while (current != null) { 
            next = current.next; 
            current.next = prev; 
            prev = current; 
            current = next; 
        } 
        node = prev; 
        return node; 
    } 
	
*--------------------------------------------------------*--------------------------------------------------------------------------------------*

2. Remove Duplicates from a Linked List

  public static LinkedListNode removeDuplicates(LinkedListNode head){
    if (head == null) {
      return head;
    }

    HashSet<Integer> dupSet = new HashSet<Integer>();
    LinkedListNode curr = head;
    dupSet.add(curr.data);

    while (curr.next != null) {
      if (!dupSet.contains(curr.next.data)) {
        // Element not found in map, let's add it.
        dupSet.add(curr.next.data);
        curr = curr.next;
      } 
      else {
        // Duplicate node found. Let's remove it from the list.
        curr.next = curr.next.next;
      }
    }
    return head;
  }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

3. Delete All Occurrences of a Given Key in a Linked List

  public static LinkedListNode deleteNode(LinkedListNode head, int key) 
  {
    LinkedListNode prev = null;
    LinkedListNode current = head;

    while (current != null) {
      if (current.data == key) {   
          if(current == head){
              head = head.next;
              current = head;
            }
        else{
            prev.next = current.next;
            current = current.next;
          }
      }
      else {
          prev = current;
          current = current.next;
      }
    }

    return head;
  }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

4. Intersection Point of Two Lists

  public static LinkedListNode intersect(LinkedListNode head1, LinkedListNode head2) {

    LinkedListNode list1node = null;
    int list1length = get_length(head1);
    LinkedListNode list2node = null;
    int list2length = get_length(head2);

    int length_difference = 0;
    if(list1length >= list2length) {
      length_difference = list1length - list2length;
      list1node = head1;
      list2node = head2;
    } else {
      length_difference = list2length - list1length;
      list1node = head2;
      list2node = head1;
    }

    while(length_difference > 0) {
      list1node = list1node.next;
      length_difference--;
    }

    while(list1node != null) {
      if(list1node == list2node) {
        return list1node;
      }

      list1node = list1node.next;
      list2node = list2node.next;
    }
    return null;
  }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

5. MERGE Two Sorted Linked List

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		
          ListNode dummy = new ListNode(0);
          ListNode head = dummy;
          
          while(l1!=null && l2!=null){
              if(l1.val < l2.val){
                 dummy.next = l1;
                 dummy = dummy.next;
                 l1 = l1.next;
              }else{
                 dummy.next = l2;
                 dummy = dummy.next;
                 l2 = l2.next; 
              }
          }
        
          dummy.next = (l1!=null) ? l1 : l2;
          
          return head.next;
          
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*

5. Sort Linked List (Merge Sort)

    public node mergeSort(node h){ 
        // Base case : if head is null 
        if (h == null || h.next == null) { 
            return h; 
        } 
  
        // get the middle of the list 
        node middle = getMiddle(h); 
        node nextofmiddle = middle.next; 
  
        // set the next of middle node to null 
        middle.next = null; 
  
        // Apply mergeSort on left list 
        node left = mergeSort(h); 
  
        // Apply mergeSort on right list 
        node right = mergeSort(nextofmiddle); 
  
        // Merge the left and right lists 
        node sortedlist = sortedMerge(left, right);    // we can use the above method also.
        return sortedlist; 
    } 

    public node getMiddle(node head){ 
        if (head == null) 
            return head; 
  
        node slow = head, fast = head; 
  
        while (fast.next != null && fast.next.next != null) { 
            slow = slow.next; 
            fast = fast.next.next; 
        } 
        return slow; 
    } 

    public node sortedMerge(node a, node b){ 
        node result = null; 
        /* Base cases */
        if (a == null) 
            return b; 
        if (b == null) 
            return a; 
  
        /* Pick either a or b, and recur */
        if (a.val <= b.val) { 
            result = a; 
            result.next = sortedMerge(a.next, b); 
        } 
        else { 
            result = b; 
            result.next = sortedMerge(a, b.next); 
        } 
        return result; 
    } 

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
6. K reverse linked list
 
    public ListNode reverseList(ListNode A, int B) {
        
        int n = B;
        ListNode curr = A, prev = null, next = null;
        
        while(curr!=null && n>0){
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
            n--;
        }
        if(curr!=null)
            A.next = reverseList(curr, B);
        
        return prev;
    }


*--------------------------------------------------------*--------------------------------------------------------------------------------------*
7. Add Two Numbers as Lists

    public ListNode addTwoNumbers(ListNode A, ListNode B) {
        int c = 0;
        ListNode dummy =new ListNode(0);
        ListNode head = dummy;
        
        while(A!=null || B!=null){
            
            int a = (A!=null) ? A.val: 0;
            int b = (B!=null) ? B.val: 0;
            int sum =  a + b + c;
            c = sum/10;
            ListNode newNode =new  ListNode(sum%10);
            dummy.next = newNode;
            dummy = dummy.next;
            A = (A!=null) ? A.next : A;
            B = (B!=null) ? B.next : B;
        }
        
        if(c>0){
            dummy.next = new ListNode(1);
        }
        
        return head.next;
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
8. Palindrome List

    ListNode left;
    public int lPalin(ListNode A) {
        left = A;
        return lPalinUtil(A);
    }

    public int lPalinUtil(ListNode right) {
        
        if(right == null)
           return 1;
           
        int isPalin = lPalinUtil(right.next);
        
        if(isPalin ==0)
           return 0;
           
        isPalin = (left.val == right.val) ? 1:0;
        
        left = left.next;
        
        return isPalin;
    }

*--------------------------------------------------------*--------------------------------------------------------------------------------------*
9. 430. Flatten a Multilevel Doubly Linked List

  class Solution {
    public Node flatten(Node head) {
        if( head == null) return head;
	    // Pointer
        Node curr = head; 
        while( curr!= null) {
            /* CASE 1: if no child, proceed */
            if( curr.child == null ) {
                curr = curr.next;
                continue;
            }
            /* CASE 2: got child, find the tail of the child and link it to p.next */
            Node temp = curr.child;
            // Find the tail of the child
            while( temp.next != null ) 
                temp = temp.next;
            // Connect tail with p.next, if it is not null
            temp.next = curr.next;  
            if( curr.next != null )  curr.next.prev = temp;
            // Connect p with p.child, and remove p.child
            curr.next = curr.child; 
            curr.child.prev = curr;
            curr.child = null;
        }
        return head;
    }
  }
--------------------------------------------------------------------------------

//avoid this soln,, try above iterative
    public Node flatten(Node head) {
        Node p = head; 
        // Traverse the list
        while (p != null) {
            if (p.child != null) {
                Node right = p.next; 
                
                //Process child
                p.next = flatten(p.child);
                p.next.prev = p;
                p.child = null; 
                         
                while (p.next != null)
                    p = p.next;
                
                //Reconnect next 
                if (right != null) { 
                    p.next = right;
                    p.next.prev = p; 
                }
            }
            p = p.next;
        }
        return head; 
    }
*--------------------------------------------------------*--------------------------------------------------------------------------------------*
