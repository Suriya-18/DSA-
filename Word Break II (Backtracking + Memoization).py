def wordBreak(s, wordDict):
    word_set = set(wordDict)
    memo = {}

    def backtrack(start):
        if start == len(s):
            return [""]
        if start in memo:
            return memo[start]

        res = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for sub in backtrack(end):
                    res.append(word + (" " if sub else "") + sub)
        memo[start] = res
        return res

    return backtrack(0)
