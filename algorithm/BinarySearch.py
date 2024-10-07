from typing import Optional
import unittest

def BinarySearch(nums : list[int] , target : int)->int:
    start : int = 0
    end : int = len(nums) - 1
    index = __binarySearch(nums,target,start,end)
    return index 
    
    
def __binarySearch(nums : list[int],target :int ,start : int , end : int)->int:
    mid : int = start + (end - start) // 2
    if start > end:
        return None    
    if nums[mid] == target:
        return mid
    elif target < nums[mid]:
        arr = __binarySearch(nums , target ,start , mid - 1)
        return arr
    elif target > nums[mid]:
        arr =__binarySearch(nums,target,mid+1,end)
        return arr
        

        
class TestBinarySearch(unittest.TestCase):
    @staticmethod
    def Test_BinarySearch():
        vec : list[int] = [1,2,3,4,5]
        target : int = 5
        index : int = BinarySearch(vec,target)
        print("index is {}".format(index))
        assert index == 4
    
if __name__ == "__main__":
    TestBinarySearch.Test_BinarySearch()