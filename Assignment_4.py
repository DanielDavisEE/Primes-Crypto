import math, functools, time, random
import my_primes, my_maths

def mul_inv(a, n):
    while math.gcd(a, n) != 1:
        return None
    base = n
    prev_x, x = 1, 0
    prev_y, y = 0, 1
    
    while n:
        q = a // n
        x, prev_x = prev_x - q * x, x
        y, prev_y = prev_y - q * y, y
        a, n = n, a % n
    return prev_x % base

def check_generator(g, factors, base):
    for f in factors:
        if pow(g, f, base) == 1:
            break
    else:
        return True
    
    return False

def find_generator(base, rand=False):
    """Assumes the base is a prime and thus the group is cyclic"""
    n = base - 1
    factors = pollard_factorise(n)
    factors.sort()
    factors = factors[1:-1]
    #print(factors)
    notFound = True
    g = 2 if not rand else random.randint(2, n)
    while notFound:
        g = g + 1 if not rand else random.randint(2, n)
        for f in factors:
            #print(g, f, base, pow(g, f, base))
            if pow(g, f, base) == 1:
                break
        else:
            notFound = False
    return g

def log(a, b, n):
    x = 0
    while True:
        if pow(a, x, n) == b % n:
            break
        x += 1
        if x == n:
            return None
    return x

def chinese_remainder(n, a):
    """n is the factors of the base, or the sub bases for the relations,
        a is the congruences of the solution in the bases of n
    """
    sum = 0
    prod = functools.reduce(lambda a, b: a*b, n)
 
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        n_inv = mul_inv(p, n_i)
        #print(a_i, n_inv, p, n_i)
        sum += a_i * n_inv * p
        #print(sum)
    #print(sum, prod, sum % prod)
    return sum % prod


def P_H_alg(a, b, p, q):
    #print(q)
    N = functools.reduce(lambda a, b: a*b , q)
    x_list = []
    q.sort()
    for i, factor in enumerate(q[:-1]):
        if q[i] == q[i+1]:
            q[i], q[i+1] = q[i] ** 2, None
    q = [x for x in q if x is not None]
    print(q)
    #1
    for i, q_i in enumerate(q):
        a_i, b_i = pow(a, N // q_i, p), pow(b, N // q_i, p)
        #2
        print(a_i, b_i)
        x_list.append(log(a_i, b_i, p))
        print(x_list)
    #3
    print(q, x_list)
    x = chinese_remainder(q, x_list)
    #print(x)
    #4
    
    #actual_log = log(a, b, p)
    #print(actual_log)
    #assert actual_log == x
    return x
    

def miller_rabin_check(n, b):
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
    
    return check_num(b)


#@simplify
def pollard_factorise(num, base=5):
    
    factors = []

    # Check against small prime factors, ensures not all factors are B-powersmooth
    primes = my_primes.prime_list(base)
    for p in primes:
        while num % p == 0:
            factors.append(p)
            num //= p
    
    def factorise(n, B):
        
        factors = []
        
        if my_primes.miller_rabin_test(n):
            factors.extend([n])
            return factors
    
        primes = my_primes.prime_list(B)
        
        q = [x ** int(math.log(B, x)) for x in primes]
        M = functools.reduce(lambda a, b: a*b , q)
        
        # Find a comprime to n, can be fixed
        a = 2
        while math.gcd(a, n) != 1:
            a += 1
        
        g = math.gcd(pow(a, M, n) - 1, n)
        
        # if 1 < g < n, g is a factor of n. Add g to factors and continue.
        if 1 < g < n:
            factors.extend(factorise(n // g, B))
            factors.extend([g])
            return factors
        
        # If g == 1, no factors p for which p-1 is B-powersmooth. Restart with 
        #   higher B
        elif g == 1:
            factors.extend(factorise(n, my_primes.next_prime(B)))
            return factors
        
        # If g == n, most likely all factors were B-powersmooth (should never be the 
        #   case). Restart with lower B
        else:
            factors.extend(factorise(n, primes[-2]))
            return factors
    
    factors.extend(factorise(num, base))
    return factors
    

    
if __name__ == "__main__":
    q1 = False
    q2 = False
    q3 = False
    q4 = False
    
    if q1:
        print(f'\tQuestion 1')
        q = [2, 3, 5, 7, 11]
        p = functools.reduce(lambda a, b: a*b , q) + 1
        print(p, my_primes.isPrime(p))
        a = find_generator(p, True)
        print(a)
        find_logs = [7, 11]
        for b in find_logs:
            x = P_H_alg(a, b, p, q)
            print(f'{x} = log_{a}({b}) mod {p}')
            
    if q2:
        print(f'\n\tQuestion 2')
        found = False
        i = 1
        limit = 10 ** 12
        while not found and i < limit:
            if my_primes.miller_rabin_test(i):
                continue
            
            for base in [2, 7, 53]:
                if i % base == 0 or not miller_rabin_check(i, base):
                    break
            else:
                print(i)
                found = True
            i += 2
                
            if (i + 1) % (limit // 100000) == 0:
                print('.', end=' ')
            if (i + 1) % (limit // 1000) == 0:
                print('\n', (i + 1) // (limit // 1000), end=' ')
    
    if q3:
        print(f'\n\tQuestion 3')
        test_funcs = [
            my_primes.isPrime,
            my_primes.trial_division,
            my_primes.miller_rabin_test,
            my_primes.fermat_test
            ]
        p = 2 ** 32 + 15
        for func in test_funcs:
            print(func(p, 1))
        factors = pollard_factorise(p - 1)
        factors.sort()
        
        print(f'3 is a primitive root of {p}: {check_generator(3, factors, p)}')
        print(f'2 is a primitive root of {p}: {check_generator(2, factors, p)}')
        
        print(factors)
        for i in range(16):
            print(f'2^32 + {i}: 3 ** {P_H_alg(3, 2 ** 32 + i, p, factors.copy())} = {2 ** 32 + i}')
    
    if q4:
        print(f'\n\tQuestion 4')
        print(f'Factors of 8927393: {pollard_factorise(8927393, 5)}')#3863, 2311