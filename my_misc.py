import time
import math

def variations(num_str):
    num_dict = {}
    for n in num_str:
        num_dict[n] = num_dict.get(n, 0)
        num_dict[n] += 1
    total = factorial(len(num_str))
    denominator = 1
    for x in num_dict.values():
        if x > 1:
            denominator *= factorial(x)
            
    return total // denominator

def compare_times(func1, func2, inputs1, inputs2, iterations):
    time1 = time.time()
    
    for x in range(iterations):
        func1(*inputs1)
    
    time2 = time.time()
    
    for x in range(iterations):
        func2(*inputs2)    

    return '{} ms, {} ms'.format(round(1000 * (time2 - time1)), round(1000 * (time.time() - time2)))
    
def binary_search(item, data):
    i, j = 0, len(data) - 1
    temp = (i + j) // 2
    while i < j:
        if data[temp] < item:
            i = temp + 1
        else:
            j = temp
        temp = (i + j) // 2
    return True if data[temp] == item else False    


def findDivisors(a):
    '''220: 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110'''
    prime_factors = prime_factor_dict(a)
    prime_factors_keys = list(prime_factors.keys())
    prime_factors_values = list(prime_factors.values())
    
    num_divisiors = 1
    for value in prime_factors_values:
        num_divisiors *= value + 1    
    
    divisors = [1] * num_divisiors
    
    prime_factors_num = len(prime_factors_keys)
    list_a = [0] * prime_factors_num
    try:
        list_a[0] = 1
    except IndexError:
        return divisors
    
    for n in range(1, num_divisiors):
        if not list_a:
            print('Too many loops')
        for i in range(prime_factors_num):
            divisors[n] *= prime_factors_keys[i] ** list_a[i]
        list_a = factor_count_increment(list_a, prime_factors_values)

    return divisors

def findDivisors2(num):
    '''220: 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110'''
    divisors = [1, num] if num not in [0, 1] else [1]
    i, temp = 2, num
    while i * i <= temp:
        if temp % i == 0:
            divisors.append(i)
            recip = temp // i
            if recip * recip != num:
                divisors.append(recip)
        i += 1
    return sorted(divisors)

def binary_increment(input_num):
    n = len(input_num)
    input_num[n - 1] += 1
    i = n - 1
    while 2 in input_num:
        input_num[i] = 0
        try:
            input_num[i - 1] += 1
        except IndexError:
            input_num.append(1)
        i -= 1
    return input_num


def factor_count_increment(input_num, max_values):
    n = len(input_num)
    input_num[0] += 1
    i = 0
    for i in range(n):
        if input_num[i] > max_values[i]:
            input_num[i] = 0
            try:
                input_num[i + 1] += 1
            except IndexError:
                return False        

    return input_num

isPalindrome = lambda word: str(word)[:len(str(word))//2] == str(word)[:(len(str(word))-1)//2:-1]

def join_nums(nums):
    return int(''.join([str(x) for x in nums]))


def timer(func, input, reps):
    initial_time = time.time()
    for _ in range(reps):
        result = func(input)
    return result, round((time.time() - initial_time) * 1000)