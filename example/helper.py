
# For linked list
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
class LinkedList(object):
    
    def printList(self,head):
        """
        Traverse every nodes in the singly linked list that is defined by this head
        For each node we encounter, print the value
        """
        curr = head
        while curr:    # equal to say: while curr is not None
            print("({} .-)->".format(curr.val), end='')
            curr = curr.next
        print("NULL")

        
# For tree

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class BinaryTree(object):
    def BFS_traversal(self, root):
        # Time Complexity: O(n)
        # Space Complexity: O(len(curr_level)+len(post_level)), O(n) at worst
        if not root:  # if root is None
            return []
        
        results = [] # store the total results
        result_level = [] # store the result of the current level
        
        curr_level = [root]
        post_level = []
        
        while curr_level:  # if curr_level is not None
            head = curr_level.pop(0)
            if not head:
                result_level.append(" ")
            else:                
                result_level.append(head.val)
                
                if head.left: # if head.left is not None
                    post_level.append(head.left)
                else:
                    post_level.append(None)
                
                if head.right: # if head.right is not None:
                    post_level.append(head.right)
                else:
                    post_level.append(None)            
            
            if not curr_level:  # if curr_level is None
                results.append(result_level)
                if post_level and self.not_all_none_check(post_level) :  # if post_level is not None
                    curr_level = post_level
                    post_level = []
                    result_level = []
                    
        return results
    
    def not_all_none_check(self, lst):
        if not lst:
            return False
        
        for val in lst:
            if val is not None:
                return True
            
        return False