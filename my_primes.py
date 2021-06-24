import time
import math
import random
import matplotlib.pyplot as plt
import my_misc

""" ######## SECURITY FUNCTIONS ######## """

def next_prime(p):
    if p % 2 == 0:
        next_p = p + 1
    else:
        next_p = p + 2
    
    while not isPrime(next_p):
        next_p += 2
    return next_p

def build_prime_list(limit, primes=[]):
    if limit <= 1:
        return []
    elif limit <= 2:
        return [2]
    elif limit <= 3:
        return [2, 3]
    
    if len(primes) < 2:
        primes = [2, 3]
        
    num, w = 5, 2
    
    while num <= limit:
        primes_iter = iter(primes)
        i = 2
        max_divisor = int(num ** 0.5)
        while i <= max_divisor:
            i = next(primes_iter)
            if num % i == 0:
                break
        else:
            primes.append(num)
        num += w
        w = 6 - w
    return primes

def sieve_of_eratosthenes(num):
    """Uses the sieve of eratosthenes to check primality with the n ** 2 
    optimisation.
    Until the result is returned, the numbers are store as bool values to save memory"""
    numbers = [0, 0] + [1] * (num - 1)
    for i in range(2, int(len(numbers) ** 0.5) + 1):
        if numbers[i]:
            for j in range(i * i, len(numbers), i):
                numbers[j] = 0
    return [i for i, x in enumerate(numbers) if x]
    
def prime_list(limit, primes=[]):
    if len(primes) < 100:
        return sieve_of_eratosthenes(limit)
    else:
        return build_prime_list(limit, primes)


def isPrimeLow(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    w = 2

    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True  

def isPrimeHigh(n, max_p):
    #check divisibility of n against known primes
    primes = prime_list(max_p)
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif any(n % p == 0 for p in primes):
        return False
    
    #multiply known primes to get c#
    primorial = 1
    for p in primes:
        primorial *= p
        
    #find all i such that gcd(i, c#) = 1, i < c#
    i_list = [x for x in range(1, primorial) if math.gcd(primorial, x) == 1]
    
    #increment c# multiplier, k, until (c#k + i) <= n ** 0.5
    k = 0
    max_divisor = n ** 0.5
    while primorial * k + i_list[-1] <= max_divisor:
        #iterate over i_list and check modularity of n in n % (c#k + i) == 0
        for i in i_list:
            divisor = primorial * k + i
            if divisor == 1:
                continue
            if n % divisor == 0:
                return False
        k += 1
    return True
        
def isPrime(n, max_p=3):
    """Just chooses between the set primorial of 6 or the changeable primorial
    Should have it choose different max_p values based on size of n
    """
    if max_p > 3:
        return isPrimeHigh(n, max_p)
    else:
        return isPrimeLow(n)

def trial_division(n, limit=None):
    """Checks a number against all primes upto limit, inclusive"""
    if limit == None:
        limit = int(n ** 0.5)
        
    primes = prime_list(limit)
    for prime in primes:
        if n < prime * prime:
            return True
        if n % prime == 0:
            return False
    return True

""" ######## DEVELOPMENT ######## """

def compact_prime_list(limit):
    # Use c# = 30, create an unpack function as well. maybe within this function?
    length = ((limit - 1) // 30 + 1) * 8
    num_list = [0] + [1] * (length - 1)
    coprimes = [1, 7, 11, 13, 17, 19, 23, 29]
    index_to_num = lambda i: 30 * (i // 8) + coprimes[i % 8]
    num_to_index = lambda n: (n - n % 30) // 30 * 8 + coprimes.index(n % 30)
    for i, x in enumerate(num_list):
        num = index_to_num(i)
        if not isPrime(num):
            num_list[i] = 0

    return num_list

def compact_prime_list2(limit):
    # Use c# = 30, create an unpack function as well. maybe within this function?
    length = ((limit - 1) // 30 + 1) * 8
    num_list = [0] + [1] * (length - 1)
    coprimes = [1, 7, 11, 13, 17, 19, 23, 29]
    index_to_num = lambda i: 30 * (i // 8) + coprimes[i % 8]
    num_to_index = lambda n: (n - n % 30) // 30 * 8 + coprimes.index(n % 30)
    '''for i, x in enumerate(num_list):
        num = index_to_num(i)
        if not isPrime(num):
            num_list[i] = 0
            '''
    max_num = index_to_num(length - 1)
    upper_lim = max_num ** 0.5
    num, i = 7, 1
    while num <= upper_lim:
        mult, j = 7, 1
        max_mult = max_num // num
        #something to do with remainders in mod 8?
        while mult <= max_mult:
            index = num_to_index(num * mult)
            num_list[index] = 0
            j += 1
            mult = index_to_num(j)
        i += 1
        num = index_to_num(i)
    return num_list

def unpack_prime_list(num_list):
    primes = [2, 3, 5]
    coprimes = [1, 7, 11, 13, 17, 19, 23, 29]
    index_to_num = lambda i: 30 * (i // 8) + coprimes[i % 8]
    for i, x in enumerate(num_list):
        if x:
            primes.append(index_to_num(i)) 
    return primes

def fermat_test(n, k=10):
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if pow(a, n - 1, n) != 1:
            return False
    return True

def miller_rabin_test(n, k=40):
    """Performs k iterations of Miller Rabin with random values of 'a'
    Deterministic for n < 341550071728321"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if not trial_division(n, 257):
        return False
    
    # Set up components
    s, d = 0, n - 1
    
    while d % 2 == 0:
        d >>= 1
        s += 1
    
    def check_num(a):
        """Performs one iteration of Miller Rabin"""
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(1, s):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False
    
    # Perform 
    small_primes = [2, 3, 5, 7, 11, 13, 17] 
    limits = [(1373653, []), (25326001, []), (118670087467, [3215031751]), \
              (2152302898747, []), (3474749660383, []), (341550071728321, [])]
    
    for i, limit in enumerate(limits):
        if n < limit[0]:
            for exception in limit[1]:
                if n == exception:
                    return False
            return all(check_num(a) for a in small_primes[:i + 2])
    
    # Perform check k times with different values of a
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if not check_num(a):
            return False
    return True

def fast_deterministic():
    pass

def ecc():
    pass
    
""" ######## TEST FUNCTIONS ######## """
'''
def test_prime_functions(min_power=0, max_power=6, power_multiplier=0.2, primorial_max=7):
    # Initialise primes list for different primorial values
    primes = prime_list(primorial_max)
    
    # Initialise dict to store prime values vs time measurements
    time_dict = {}
    for p in primes:
        time_dict[p] = []

    min_power_int = int(min_power / power_multiplier) # Min power of 10 used in primality testing
    max_power_int = int(max_power / power_multiplier) # Max power of 10 used in primality testing
    test_num = 20 # number of times to check each number for primality
    
    for max_p in primes: # iterate of different values of primorial
        
        for power in range(min_power_int, max_power_int): # iterate over different numbers to check primality of
            exponent = round(power_multiplier * power * 10) / 10
            num = int(10 ** exponent)
        #for num in range(min_num + 1, max_num, 24):
            time1 = time.time() # start timing
            #print(num)
            for x in range(round(2 ** (exponent // 2) + power * 20 + 100)): # check each number 'test_num' times
                primes1 = prime_list(max_p)
                isPrime4(num + x, primes1) # check primality. stuck here, max_p = 5, num = 1
            time_total = time.time() - time1 # finish timing
            time_dict[max_p].append(time_total * 100 / round(2 ** (exponent // 2) + power * 10))
    
    x = [10 ** (power_multiplier * n) for n in range(min_power_int, max_power_int)]
    #x = list(range(min_num + 1, max_num, 24))
    labels = []
    for label, t in time_dict.items():
        #labels.append(t, label)
        y = [round(d * 1000) for d in t]
        plt.plot(x, y, label=str(label))
    plt.legend()
    plt.xlabel('Number Size')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def test_prime_functions2(min_power=0, max_power=5, primorial_max=7):
    # Initialise primes list for different primorial values
    primes = prime_list(primorial_max)
    
    # Initialise dict to store prime values vs time measurements
    time_dict = {}
    for p in primes:
        time_dict[p] = []

    min_num = 10 ** min_power # Min power of 10 used in primality testing
    max_num = 10 ** max_power # Max power of 10 used in primality testing
    test_num = 50 # number of times to check each number for primality
    print('started')
    available_primes = [p for p in prime_list(max_num) if p >= min_num]
    print('done', len(available_primes))
    for max_p in primes: # iterate of different values of primorial
        
        for p in available_primes: # iterate over different numbers to check primality of

            time1 = time.time() # start timing
            #print(num)
            for x in range(test_num): # check each number 'test_num' times
                primes1 = prime_list(max_p)
                isPrime4(p + 2 * x, primes1) # check primality. stuck here, max_p = 5, num = 1
            time_total = time.time() - time1 # finish timing
            time_dict[max_p].append(time_total)
        print(max_p)
    x = available_primes#[10 ** (power_multiplier * n) for n in range(min_power_int, max_power_int)]
    #x = list(range(min_num + 1, max_num, 24))
    labels = []
    for label, t in time_dict.items():
        
        y = [round(d * 1000) for d in t]
        print(y)
        plt.plot(x, y, label=str(label))
    plt.legend()
    plt.xlabel('Number Size')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
def test_prime_functions3(primorial_min=2, primorial_max=9, iterations=1, min_digits=1, max_digits=20):
    # Initialise primes list for different primorial values
    primes = prime_list(primorial_max)
    p_index = primes.index(primorial_min)
    
    test_primes = {1: 7, # 3
                  2: 23, # 3
                  3: 101, # 3
                  4: 1009, # 3
                  5: 10007, # 3
                  6: 100003, # 3
                  7: 1000003, # 5
                  8: 10000019, # 7
                  9: 100000007, # 7
                  10: 1000000007, # 7
                  11: 10000000019, # 11
                  12: 100000000003, # 11
                  13: 1000000000039, # 11
                  14: 10083087720779, # 11 / 13
                  15: 112272535095293, # 13
                  16: 1003026954441971, # 13
                  17: 10022390619214807, # 17
                  18: 100055128505716009, # 17
                  19: 1024628340621757567, # 19
                  20: 10168938831019335571, # 19
                  21: 122734414681536249703} # 
    # 0.0297 * x ** 2 + 0.3124 * x + 1.7526
    
    # Initialise dict to store prime values vs time measurements
    time_dict = {}
    for p in [x for x in test_primes.keys() if min_digits <= x <= max_digits]:
        time_dict[p] = []
    
    for i, prime in [x for x in test_primes.items() if min_digits <= x[0] <= max_digits]:
        print('{}, {} iterations'.format(prime, iterations))
        for p in primes[p_index + 1:]:
            time1 = time.time()
            
            for _ in range(iterations):
                primality = isPrimeAll(prime, p)
                
            time_final = round(1000 * (time.time() - time1))
            time_dict[i].append(time_final)  
    
            print('{}: {} ms'.format(p, time_final))
    
    for label, t in time_dict.items():
        
        y = [round(d * 1000) for d in t]
        plt.plot(primes[p_index + 1:], t, label=str(label))
    plt.legend()
    plt.xlabel('Number Size')
    plt.ylabel('Time (ms)')
    #plt.yscale('log')
    plt.show()

def test_prime_functions4(digit_count, iterations=1):
    primes = prime_list(20)
    times = []
    prime_dict = {1: 7,
                  2: 23,
                  3: 101,
                  4: 1009,
                  5: 10007,
                  6: 100003,
                  7: 1000003,
                  8: 10000019,
                  9: 100000007,
                  10: 1000000007,
                  11: 10000000019,
                  12: 100000000003,
                  13: 1000000000039,
                  14: 10083087720779,
                  15: 112272535095293,
                  16: 1003026954441971,
                  17: 10022390619214807,
                  18: 100055128505716009,
                  19: 1024628340621757567,
                  20: 10168938831019335571}

    prime = prime_dict[digit_count]
    print('  Prime: {}, Digits: {}'.format(prime, digit_count))
    
    for p in primes[1:]:
        print('{}:'.format(p), end=' ')
        time1 = time.time()
        
        for _ in range(iterations):
            primality = isPrimeAll(prime, p)
            
        time_final = round(1000 * (time.time() - time1))
        print(primality, time_final, 'ms')
        times.append(time_final)
    
    plt.plot(primes[1:], times, 'bo')'''
    
'''######## REDUNDANT ########

def isPrime(num, primes):
    """Primes is a list of all prime numbers upto num"""
    primes_iter = iter(primes)
    i = 2
    
    if num > 1:
        while i * i <= num:
            i = next(primes_iter)
            if num % i == 0:
                return False
        return True
    return False
    
def isPrime2(num):
    """Use isPrime3 instead"""
    primes = []
    i = 2
    if num < 2:
        return False
    while i * i < num + 1:
        if isPrime(i, primes):
            primes.append(i)
            if num % i == 0:
                return False
        i += 1
    return True
    

def sieve_of_eratosthenes(num):
    """Uses the sieve of erotosthenes to check primality with the n ** 2 
    optimisation.
    Until the result is returned, the numbers are store as bool values to save memory
    >>Bad Implementation"""
    numbers = [0] + [1] * (num - 1)
    i = 2
    index = i * i - 1
    while i * i <= num:
        if numbers[i - 1] == 1:
            while index < num:
                if numbers[index] == 1:
                    numbers[index] = 0
                index += i
        i += 1
        index = i * i - 1
    return [i + 1 for i, x in enumerate(numbers) if x]

'''