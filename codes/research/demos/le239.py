#!/user/bin/env python
# coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : le239.py
#@time   : 2023-10-16 10:32:30
'''
from collections import deque


class Solution:
    def printq(self, q):

        n = len(q)
        tmp = []
        s = ""
        for i in range(n):
            _l = q.popleft()
            s = s + str(_l) + ","

            q.append(_l)

        print("queue:", s)

    def maxSlidingWindow(self, nums, k: int):
        q = deque()
        ret = []

        for i, item in enumerate(nums):
            if i == 3:
                print("check")
            print("begin", i, nums[i])
            self.printq(q)
            if i >= k:
                # 加进去结果
                _left = q.popleft()
                ret.append(_left)
                q.appendleft(_left)

                # 维护窗口
                _left = q.popleft()
                if _left != nums[i - k]:
                    q.appendleft(_left)
                # else:
                #     print("pop out of window:", _left)
            if not len(q):
                q.append(nums[i])
                continue
            else:
                _right = q.pop()  # ??????????????????????
                if _right >= item:
                    print("run here,", _right)
                    q.append(_right)
                    q.append(item)
                else:
                    while _right < item and len(q):
                        _right = q.pop()
                        # print("poping from right that smaller:",_right)
                    if _right >= item:
                        q.append(_right)

                    q.append(item)

        if len(q):
            ret.append(q.popleft())
        return ret


s = Solution()
ret = s.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3)
print(ret)
