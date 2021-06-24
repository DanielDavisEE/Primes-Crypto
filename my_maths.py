import math, time, random
import sys
import matplotlib.pyplot as plt
import numpy
import my_primes
#sys.setrecursionlimit(1000000)

digital_root = lambda x: sum(int(n) for n in str(x))

def my_sqrt(num, margin_of_error=0.0001):
    """Works, but just use x ** 0.5, its faster"""
    if num == 0:
        return 0
    else:
        exponent = 0
        initial_guess = 0
        
        while num < 0:
            num *= 10 ** 2
            exponent -= 1
            
        while num > 100:
            num /= 10 ** 2
            exponent += 1
                
        if num <= 10:
            initial_guess = 2
        else:
            initial_guess = 6
            
    x = initial_guess
    while abs(num - x * x) > margin_of_error:
        x = (num / x + x) / 2
        
    return x * 10 ** exponent


def product_of(num_list):
    result = 1
    for num in num_list:
        result *= num
    return result


def factorial(num):
    total = 1
    for i in range(2, num + 1):
        total *= i
    return total

def my_gcd(a, b):
    """just use math.gcd(), it's the same but written in C"""
    while b:
        a, b = b, a % b
    return a

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

"""Not sure where this is used in other code"""
def mult_inverse(a, n): return mul_inv(a, n)

def jacobi_symbol(a, n):
    a = a % n
    s = 1
    if n == 1:
        return 1
    if a == 0 or n % 2 == 0:
        return 0
    while a % 2 == 0:
        a >>= 1
        s *= (-1) ** ((n * n - 1) // 8)
        if s == 0:
            return 0
    if a == 1:
        s*= 1
    elif math.gcd(a, n) != 1:
        return 0
    else:
        i = 1
        if n % 4 == 3 and a % 4 == 3:
            i = -1
        s *= i * jacobi_symbol(n, a)
    return s

def test_jacobi(a, b):
    for i in range(1, a, 2):
        for j in range(b):
            print('{:2}'.format(jacobi_symbol(j, i)), end=' ')
        print()
        
def pi_approximator(lim):
    moe = 0.00001
    j = 1
    print(math.pi)
    j_list = []
    while j < 5000:
        for i in range(j * 3, j * 4):
            if math.gcd(i, j) != 1:
                continue
            approximation = i / j
            if i / j > 3.1416:
                continue
            if abs(math.pi - approximation) < moe:
                
                print('{} / {} = {} | diff = {}'.format(i, j, approximation, abs(math.pi - approximation))) #
                j_list.append(j)
        j += 1
    mod = 500
    x = numpy.array([pow(n, 1, mod) for n in j_list])
    y = numpy.array([int(n / mod) for n in j_list])
    
    plt.set_cmap('viridis')    
    plt.pcolormesh(x, y)
    plt.show()
        
def prime_factorise(num):
    limit = num
    potential_factors = my_primes.prime_list(limit)
    factors = {}
    i = 0
    while num > 1:
        p = potential_factors[i]
        while num % p == 0:
            num //= p
            factors[p] = factors.get(p, 0) + 1
        i += 1
    return factors

def pi_checker(pi_approximator):
    round_time = lambda t: round(1000 * t)
    def wrapper(*args):
        start_time = time.time()
        test_value = pi_approximator(*args)
        full_time = time.time() - start_time
        diff = abs(math.pi - pi_approximator(*args))
        print(f'{test_value}, {diff}, {round_time(full_time)} ms')
    return wrapper

@pi_checker
def pi_series(k=1):
    temp = 0
    for i in range(k):
        temp += ((-1) ** i * math.factorial(6 * i) * (13591409 + 545140134 * i)
                 / (math.factorial(3 * i) * (factorial(i)) ** 3 * 640320 ** (3 * i + 3/2)))
    return 1 / (12 * temp)


def factorise(num):
    limit = int(num ** 0.5) + 1
    factors = []
    [factors.extend([i, num // i]) for i in range(1, limit) if num % i == 0]
    return sorted(factors)


def find_generator(base):
    """Assumes the base is a prime and thus the group is cyclic"""
    n = base - 1
    factors = factorise(n)[1:-1]
    notFound = True
    
    while notFound:
        g = random.randint(1, base)
        for f in factors:
            if pow(g, f, base) == 1:
                break
        else:
            notFound = False
    return g

def check_generator(g, factors, base):
    for f in factors:
        if pow(g, f, base) == 1:
            break
    else:
        return True
    
    return False

def find_generators(base, limit):
    """Assumes the base is a prime and thus the group is cyclic"""
    n = base - 1
    factors = factorise(n)[1:-1]
    notFound = True
    generators = []
    for i in range(2, limit):
        if check_generator(i, factors, base):
            generators.append(i)
    return generators

def find_generator(base, rand=False):
    """Assumes the base is a prime and thus the group is cyclic"""
    n = base - 1
    factors = factorise(n)[1:-1]
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

if __name__ == '__main__':
    p = find_generators(200003, 20)
    print(p)
#a = random.randint(1, p)
#A = p ** a

#print(p, a, A)

#gens = set()
#for _ in range(1000):
    #gens.add(find_generator(31))
#print(sorted(list(gens)))


#355 / 113
#104348 / 33215