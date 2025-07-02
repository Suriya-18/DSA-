class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = ""

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_end = True
        curr.word = word

class Solution:
    def longestWord(self, words):
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        stack = [trie.root]
        longest = ""

        while stack:
            node = stack.pop()
            if node != trie.root and not node.is_end:
                continue
            if node.word:
                if len(node.word) > len(longest) or \
                   (len(node.word) == len(longest) and node.word < longest):
                    longest = node.word
            for child in node.children.values():
                stack.append(child)
        
        return longest

# Example usage
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
print(Solution().longestWord(words))  # Output: "apple"
