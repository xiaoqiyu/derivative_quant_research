#!/user/bin/env python
#coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : le904.py
#@time   : 2023-10-20 23:02:28
'''

class Solution:
    def totalFruit(self, fruits) -> int:
        l, r = 0,0
        cache ={}
        ret = 0
        while r < len(fruits):
            type_l = fruits[l]
            type_r = fruits[r]
            _c = cache.get(type_r) or 0
            _c += 1
            cache.update({type_r:_c})
            print(r, type_r, cache)
            while len(cache) > 2:
                _c1 = cache.get(fruits[l]) or 0
                cache.update({fruits[l]:_c1-1})


                if cache[fruits[l]] == 0:
                    cache.pop(fruits[l])
                print(l, fruits[l])
                l += 1
            ret = max(ret, r-l+1)
            r+= 1
            print("end of while:",r,ret)
        return ret


s=Solution()
ret = s.totalFruit([1,2,3,2,2])
print(ret)