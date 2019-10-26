def maxLen(arr):
    u = [1 for i in range(len(arr))]
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                u[i] = max(u[i], u[j] + 1)
    return max(u)


'''
最长连续递增子序列
dp[i]以nums[i]结尾的最长递增子序列的长度
if nums[i] > nums[j], 说明nums[i]能缀到nums[j]后面，
那么dp[j]就能+1了
dp[i+1] = max(dp[i + 1], dp[j] + 1)
'''


def length_of_lis(nums):
    len_nums = len(nums)
    if len_nums == 0:
        return 0

    dp = [1] * len_nums
    for i in range(len_nums - 1):
        for j in range(i + 1):
            # 如果nums[i+1]能缀在nums[j]后面的话，就dp[j]+1
            if nums[i + 1] > nums[j]:
                # 缀完的结果还要看看是不是比我大
                dp[i + 1] = max(dp[i + 1], dp[j] + 1)
    return max(dp)


def max_successive_series_len(arr):
    """
    寻找最大连续子序列，子序列的下标必须是相连的
    Parameters
    ----------
    arr

    Returns
    -------

    """
    max_area_len = 0
    for i in range(len(arr)):
        print(i)
        for j in range(i + 1, len(arr)):
            if arr[j] > arr[j - 1]:
                max_area_len = max(j - i + 1, max_area_len)
            else:
                i = j

    return max_area_len


if __name__ == '__main__':
    nums = [1, 7, 3, 5, 9, 4, 8, 20, 19]
    res = max_successive_series_len(nums)
    print(res)
