import my_maths, my_misc
import random
import math
import my_primes
import math
import itertools, re

class Diffie_Helman():
    
    def __init__(self, base=None):
        if base is None:
            base = self.find_prime()
            
        self.base = base
        self.order = base - 1
        generator = self.find_generator()
    
    def find_generator(self):
        n = self.order
        factors = my_maths.factorise(n)[1:-1]
        notFound = True
        
        while notFound:
            g = random.randint(1, self.base)
            for f in factors:
                if pow(g, f, self.base) == 1:
                    break
            else:
                notFound = False
        return g

class ECC():
    """Maybe one day I'll actually be able to fill this in..."""
    def __init__(self):
        pass

class RSA_basic():
    """An implementation ofthe RSA cryptosystem which requires the input of the 
    two primes to be used to generate the keys
    """
    def __init__(self, prime1=None, prime2=None):
        self.create_new_keys(prime1, prime2)
    
    def generate_primes(self):
        raise TypeError('RSA_basic() requires 2 primes as inputs, 0 were given')
    
    def test_small_primes(self, key):
        primes = my_primes.prime_list(257)
        for p in primes:
            while key % p == 0:
                key //= p
        return key == 1
    
    def create_new_keys(self, p, q):
        """Put in checks for strong keys
        """
        if not all([p, q]):
            p, q = self.generate_primes()
        
        self.primes = p, q        
        
        phi = (p - 1) * (q - 1)
        n, e = p * q, 0
        while math.gcd(e, phi) != 1:
            e = random.randint(2, phi)
        d = my_maths.mult_inverse(e, phi)
            
        self.max_length = n
        self.public_key = n, e
        self.private_key = d
        
        # Chinese Remainder Theorem Components
        d_p = d % (p - 1)
        d_q = d % (q - 1)
        q_inv = my_maths.mult_inverse(q, p)
        assert (q_inv * q) % p == 1
        self.c_r_t_keys = d_p, d_q, q_inv
        
        if q < p < 2 * q and d < n ** 0.25 / 3:
            self.create_new_keys()
        
    def text_to_num(self, text):
        return [ord(x) - 96 for x in text]
    
    def num_to_text(self, nums):
        return [chr(int(x) + 96) for x in nums]
        
    def encrypt(self, plaintext):
        plaintext_num = self.text_to_num(plaintext)
        i, block, count = 1, 0, 3
        num_blocks = [plaintext_num[0]]
        while i < len(plaintext_num):
            temp = num_blocks[block] + plaintext_num[i] * 10 ** count
            if temp < self.max_length:
                num_blocks[block] = temp
                i += 1
                count += 3
            else:
                num_blocks.append(plaintext_num[i])
                i += 1
                block += 1
                count = 3
        cipher = [pow(x, self.public_key[1], self.public_key[0]) for x in num_blocks]
        return cipher
    
    def decrypt(self, cipher_num, chinese_remainder=False):
        if chinese_remainder:
            num_blocks = [self.chinese_remainder_decrypt(x) for x in cipher_num]
        else:
            num_blocks = [pow(x, self.private_key, self.public_key[0]) for x in cipher_num]
        plaintext_num = []
        for num in num_blocks:
            block = str(num)
            for n in range(len(block), 0, -3):
                start = n - 3 if n - 3 > 0 else 0
                char = block[start : n]
                plaintext_num.append(char)
        plaintext = self.num_to_text(plaintext_num)
        return ''.join(plaintext)
    
    def chinese_remainder_decrypt(self, c):
        p, q = self.primes
        d_p, d_q, q_inv = self.c_r_t_keys
        m_1 = pow(c, d_p, p)
        m_2 = pow(c, d_q, q)
        h = q_inv * (m_1 - m_2) % p
        return m_2 + h * q
    
    # Cipher Cracking Functions
    
    def fermat_factorisation(num):
        pass

class RSA(RSA_basic):
    """Version of RSA which generates the primes to be used to generate the keys.
    Some checks are included to ensure stronger key generation.
    """
    def __init__(self):
        super().__init__()
        if self.max_length < 256:
            raise ValueError("Primes too small")
        
    def generate_primes(self):
        p = random.randrange(10 ** 100 + 1, 10 ** 140, 2)
        while not my_primes.miller_rabin_test(p) or self.test_small_primes(p):
            p = random.randrange(10 ** 100 + 1, 10 ** 140, 2)
        
        q = random.randrange(10 ** 100 + 1, 10 ** 140, 2)
        while not my_primes.miller_rabin_test(q) or self.test_small_primes(q) or abs(p - q) < 2 * (p * q) ** 0.25:
            q = random.randrange(10 ** 100 + 1, 10 ** 140, 2)  
            
        return p, q    
        
    def text_to_num(self, text):
        return [ord(x) for x in text]
    
    def num_to_text(self, nums):
        return [chr(int(x)) for x in nums]
    
    def test(self):
        print(self.private_key, self.public_key)
     

class Affine():
    
    def __init__(self, base=26):
        self.base = base
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.word_list = None
        
        if base == 26:
            self.invertible_nums = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
            self.error_message = "a = {} is not invertible mod {}, use one of:\n\t{}"
        else:
            self.error_message = "a = {1} is not invertible mod {2}, use a number coprime to {2}"            
        
    def encrypt_caesar(self, plaintext, a=3):
        return self.encrypt(plaintext, 1, a)
    
    def decrypt_caesar(self, cipher, a=3):
        return self.decrypt(cipher, 1, a)
        
    def rot13(self, text):
        return self.encrypt(text, 1, 13)
        
    def encrypt(self, plaintext, a=None, b=None):
        if a is None:
            a = random.choice(self.invertible_nums)
        if b is None:
            b = random.randint(0, 25)
            while a == 1 and b == 0:
                b = random.randint(0, 25)
        
        if self.base == 26 and a not in self.invertible_nums:
            raise ValueError(self.error_message.format(a, self.base, self.invertible_nums))
        elif not my_maths.mult_inverse(a, self.base):
            raise ValueError(self.error_message.format(a, self.base))   
        
        else:
            text_words = plaintext.split(' ')
            cipher = []
            inverse = my_maths.mult_inverse(a, self.base)
            for word in text_words:
                cipher_word = ''
                for l in word:
                    if not l.isalpha(): # Punctuation or numbers
                        cipher_word += l
                    elif l.isupper():
                        cipher_word += chr((a * (ord(l) - 65) + b) % self.base + 65)
                    else:
                        cipher_word += chr((a * (ord(l) - 97) + b) % self.base + 97)
                        
                cipher.append(cipher_word)
                #print(word, cipher_word)
                
            return ' '.join(cipher)
        
    def decrypt(self, cipher, a, b):
        if self.base == 26 and a not in self.invertible_nums:
            raise ValueError(self.error_message.format(a, self.base, self.invertible_nums))
        elif not my_maths.mult_inverse(a, self.base):
            raise ValueError(self.error_message.format(a, self.base))
        
        else:
            cipher_words = cipher.split(' ')
            plaintext = []
            inverse = my_maths.mult_inverse(a, self.base)
            for word in cipher_words:
                plaintext_word = ''
                for l in word:
                    if not l.isalpha(): # Punctuation or numbers
                        plaintext_word += l
                    elif l.isupper():
                        plaintext_word += chr((((ord(l) - 65) - b) * inverse) % self.base + 65)
                    else:
                        plaintext_word += chr((((ord(l) - 97) - b) * inverse) % self.base + 97)
                        
                plaintext.append(plaintext_word)
                #print(word, plaintext_word)
            return ' '.join(plaintext)
        
    def find_invertables(self):
        return [x for x in range(1, self.base) if my_maths.mult_inverse(x, self.base)]
    
    # Cipher Cracking Functions
    """Gonna need a dictionary to check against"""
    def get_word_list(self):
        with open('D:\Daniel Davis\Documents\Coding\Python\Typing Project\\english_word_frequency_counts.txt', 'r', encoding='utf-8-sig') as infile:
            words = infile.readlines()
            words = {w.split(',')[0] for w in words}
        
        self.word_list = words
    
    def brute_force(self, cipher):
        if self.word_list is None:
            self.get_word_list()
            
        options = []
        for a in self.invertible_nums:
            for b in range(0, 26):
                option = self.decrypt(cipher, a, b)
                prob = self.judge_plaintext(option)
                options.append((option, (a, b), prob))
            
        options.sort(key=lambda x: x[2])
        options.reverse()
        return options
    
    def brute_force_caesar(self, cipher):
        if self.word_list is None:
            self.get_word_list()
            
        options = []
        for n in range(1, 26):
            option = self.decrypt_caesar(cipher, n)
            prob = self.judge_plaintext(option)
            options.append((option, n, prob))
            
        options.sort(key=lambda x: x[2])
        options.reverse()
        return options
    
    def frequency_analysis(self, cipher, frequency_list=None):
        """Performs a frequency analysis on a cipher to decipher it. Needs more
           lenience to be useful as the correct options are ignored. To find the
           correct answer the letter frequency must be in the correct order."""
        alphabet_length = len(self.alphabet)
        
        if self.word_list is None:
            self.get_word_list()
        
        if not frequency_list:
            frequency_list = list('etaoinsrhdlucmfywgpbvkxqjz')
        
        letter_counts = [0] * alphabet_length
        for letter in cipher:
            if letter.isupper():
                index = ord(letter) - 65
                letter_counts[index] += 1
            if letter.islower():
                index = ord(letter) - 97
                letter_counts[index] += 1
        
        letter_to_ascii = lambda x: ord(x) - 97
        ascii_to_letter = lambda x: chr(x + 97)
        
        # Produce an alphabet sorted by frequency in cipher and then in order from frequency_list
        letter_rank = list(self.alphabet)
        letter_rank.sort(key=lambda x: frequency_list.index(x))
        letter_rank.sort(key=lambda x: letter_counts[ord(x) - 97], reverse=True)
        
        
        # A dict with keys as number of occurrances and values as the letters
        count_dict = {}
        for i, count in enumerate(letter_counts):
            count_dict[count] = count_dict.get(count, []) + [chr(i+97)]
        
        # A list which gives the number of letters which have the same number of
        #    occurrances as each letter
        common_occurrence = [0] * alphabet_length
        for occurrence, letters in count_dict.items():
            for letter in letters:
                common_occurrence[ord(letter)-97] = len(letters)
        
        # A list which groups together numbers with equal number of occurrances and assigns a number     
        occurrance_group = [0] * alphabet_length
        occurrance_dict = {}
        group = 0
        current_group = letter_counts[ord(letter_rank[0])-97]
        for l in letter_rank:
            if letter_counts[ord(l)-97] != current_group:
                current_group = letter_counts[ord(l)-97]
                group += 1
                
            occurrance_group[ord(l)-97] = group
            occurrance_dict[group] = occurrance_dict.get(group, []) + [l]
        
        num_groups = group + 1
        
        # A list which gives the number of letters prior to the group of this letter,
        #    grouped by number of occurrances
        previous_letters = [0] * alphabet_length
        count = 0
        occurrence = common_occurrence[ord(letter_rank[0])-97]
        tmp = occurrence
        for i in range(len(previous_letters)):
            if tmp == 0:
                count += occurrence
                occurrence = common_occurrence[ord(letter_rank[i])-97]
                tmp = occurrence
                
            previous_letters[ord(letter_rank[i])-97] = count
            
            tmp -= 1
        
        decrypt_dict_options = []
        tmp = [0] * len(set(letter_counts))
        
        group_iterations = {x: [None, [None]]
                            for x in {n for n in occurrance_dict.keys()}}
        
        def choose_group_combinations(group_dict):
            def constant_iter(i):
                while True:
                    yield tuple(range(i-1, -1, -1))
                    
            permutation_iter = lambda i: itertools.permutations(range(i), i)
            
            test = [0] * num_groups
            num_options = [len(occurrance_dict[x]) for x in range(num_groups)]
            
            iterator_template = [(permutation_iter, num_options[x]) for x in range(num_groups)]
            
            for group in range(num_groups):
                if letter_counts[ord(occurrance_dict[group][0])-97] == 0:
                    iterator_template[group] = constant_iter, num_options[group]
                    
            #print(iterator_template)
            iterators = [iter_func(i) for iter_func, i in iterator_template]
            
            combination = [next(x) for x in iterators]
            end_conditions = [tuple(sorted(list(range(n)), reverse=True)) for n in num_options]
            while True:
                #print(test)
                yield combination
                
                for n in range(num_groups-1, -1, -1):
                    #print(combination[n], end_conditions[n])
                    if combination[n] == end_conditions[n]:
                        m = num_options[n]
                        iterators[n] = iterator_template[n][0](iterator_template[n][1])
                        combination[n] = next(iterators[n])
                        test[n] = 0
                    else:
                        test[n] += 1
                        combination[n] = next(iterators[n])
                        break
                else:
                    break

        
        combination = choose_group_combinations(group_iterations)
        check = 0
        for c in combination:
            #print(c)
            decrypt_dict = {}
            for l in range(alphabet_length):
                letter = letter_rank[l]
                i = ord(letter) - 97
                previous = previous_letters[i]
                group = occurrance_group[i]
                
                group_index = l - previous
        
                permutation = c[group][group_index]
                
                decrypt_dict[letter] = frequency_list[previous + permutation]
            decrypt_dict_options.append(decrypt_dict)
        
        # Rate and sort the possible plaintext options
        plaintext_options = self.substitute_with_dict(cipher, decrypt_dict_options)
        ranked_plaintext_options = [(x, self.judge_plaintext(x)) for x in plaintext_options]
        ranked_plaintext_options.sort(key=lambda x: x[1], reverse=True)

        return ranked_plaintext_options
        
    def substitute_with_dict(self, cipher, substitute_dict):
        if type(substitute_dict) is dict:
            substitute_dict = [substitute_dict]
        
        plaintext_options = []
        for d in substitute_dict:
            plaintext = ''
            for letter in cipher:
                if letter.isupper():
                    plaintext += d[letter.lower()].upper()
                elif letter.islower():
                    plaintext += d[letter]
                else:
                    plaintext += letter
            plaintext_options.append(plaintext)
            
        return plaintext_options
    
    def judge_plaintext(self, plaintext):
        if self.word_list is None:
            self.get_word_list()
            
        word_list = [word.lower() for word in re.split(r'[^a-zA-Z0-9\']+', plaintext) if word.isalpha()]
        prob = 0
        for word in word_list:
            if word in self.word_list:
                prob += 1
        prob /= len(word_list)
        
        return prob
    
    def find_decryption_dict(self, plaintext, ciphertext):
        decrypt_dict = {x: None for x in self.alphabet}
        i = 0
        while not all(decrypt_dict.values()):
            decrypt_dict[ciphertext[i]] = plaintext[i]
            i += 1
            if i == len(plaintext):
                break
        return decrypt_dict

    def find_encryption_dict(self, plaintext, ciphertext):
        encrypt_dict = {x: None for x in self.alphabet}
        i = 0
        while not all(decrypt_dict.values()):
            decrypt_dict[plaintext[i]] = ciphertext[i]
            i += 1
            if i == len(plaintext):
                break
        return encrypt_dict
    
    def encrypt_keyword(self, plaintext, keyword):
        # Create substitution dict
        decrypt_dict = {}
        count = 0
        for letter in keyword:
            if decrypt_dict.get(chr(count + 97), None) is None:
                decrypt_dict[chr(count + 97)] = letter
                count += 1
            
        for letter in self.alphabet:
            if letter not in decrypt_dict.values():
                decrypt_dict[chr(count + 97)] = letter
                count += 1
                           
        return self.substitute_with_dict(plaintext, decrypt_dict)[0]
    
    def decrypt_keyword(self, ciphertext, keyword):
        # Create substitution dict
        decrypt_dict = {}
        count = 0
        for letter in keyword:
            if decrypt_dict.get(letter, None) is None:
                decrypt_dict[letter] = chr(count + 97)
                count += 1
            
        for letter in self.alphabet:
            if decrypt_dict.get(letter, None) is None:
                decrypt_dict[letter] = chr(count + 97)
                count += 1
                
        return self.substitute_with_dict(ciphertext, decrypt_dict)[0]
    
    def brute_force_keyword(self, ciphertext, extra_words=None):
        if self.word_list is None:
            self.get_word_list()
            
        if extra_words is None:
            extra_words = []
        #print(len(self.word_list))
        
        options = []
        for word in extra_words:
            option = self.decrypt_keyword(ciphertext, word)
            prob = self.judge_plaintext(option)
            options.append((option, word, prob))
    
        for word in self.word_list:
            option = self.decrypt_keyword(ciphertext, word)
            prob = self.judge_plaintext(option)
            options.append((option, word, prob))        
            
        options.sort(key=lambda x: x[2])
        options.reverse()
        return options
    
    def encrypt_caesar_variant(self, plaintext, a=0, increment=1):
        a %= 26
        increment %= 26
        ciphertext = ''
        for letter in plaintext:
            if letter.isupper():
                ciphertext += chr(((ord(letter) - 65) + a) % self.base + 65)
                a += increment
            elif letter.islower():
                ciphertext += chr(((ord(letter) - 97) + a) % self.base + 97)
                a += increment
            else:
                ciphertext += letter
            
            a %= 26
            
        return ciphertext
        
    def decrypt_caesar_variant(self, ciphertext, a=0, increment=1):
        a %= 26
        increment %= 26
        plaintext = ''
        for letter in ciphertext:
            if letter.isupper():
                plaintext += chr(((ord(letter) - 65) - a) % self.base + 65)
                a += increment
            elif letter.islower():
                plaintext += chr(((ord(letter) - 97) - a) % self.base + 97)
                a += increment
            else:
                plaintext += letter
            
            a %= 26
            
        return plaintext
    
    def brute_force_caesar_variant(self, ciphertext):
        options = []
        for a in range(26):
            for increment in range(26):
                option = self.decrypt_caesar_variant(ciphertext)
                prob = self.judge_plaintext(option)
                options.append((option, (a, increment), prob))
                
        options.sort(key=lambda x: x[2], reverse=True)
        return options
    
    def incomplete_sub_dict(self, ciphertext, d):
        
        plaintext = ''
        for letter in ciphertext:
            if letter.lower() in d:
                if letter.isupper():
                    plaintext += d[letter.lower()].upper()
                else:
                    plaintext += d[letter]
            else:
                if letter.isalpha():
                    plaintext+='_'
                else:
                    plaintext += letter
        return plaintext
    
    def encrypt_vigenere(self, plaintext, keyword):
        ciphertext = ''
        nonletters = 0
        for i, letter in enumerate(plaintext):
            a = ord(keyword.lower()[(i - nonletters) % len(keyword)])-97
            if letter.isupper():
                ciphertext += chr(((ord(letter) - 65) + a) % self.base + 65)
            elif letter.islower():
                ciphertext += chr(((ord(letter) - 97) + a) % self.base + 97)
            else:
                nonletters += 1
                ciphertext += letter
        return ciphertext
    
    def decrypt_vigenere(self, ciphertext, keyword):
        plaintext = ''
        nonletters = 0
        for i, letter in enumerate(ciphertext):
            a = ord(keyword.lower()[(i - nonletters) % len(keyword)])-97
            if letter.isupper():
                plaintext += chr(((ord(letter) - 65) - a) % self.base + 65)
            elif letter.islower():
                plaintext += chr(((ord(letter) - 97) - a) % self.base + 97)
            else:
                nonletters += 1
                plaintext += letter
        return plaintext
    
    def brute_force_vigenere(self, ciphertext, extra_words=None, known_word=True):
        """Only use known_word=False if True fails, it takes a very long time 
           to run on any decently long keywords as it checks every possible 
           combination of letters upto max_keyword_length."""
        
        max_keyword_length = 5
        
        if known_word:
            if self.word_list is None:
                self.get_word_list()
                
            if extra_words is None:
                extra_words = []
            
            options = []
            for word in extra_words:
                option = self.decrypt_vigenere(ciphertext, word)
                prob = self.judge_plaintext(option)
                options.append((option, word, prob))
        
            for word in self.word_list:
                option = self.decrypt_vigenere(ciphertext, word)
                prob = self.judge_plaintext(option)
                options.append((option, word, prob))        
                
            options.sort(key=lambda x: x[2], reverse=True)
            return options
        else:
            options = []
            for i in range(1, max_keyword_length): # Word length
                for word in itertools.product(range(26), repeat=i):
                    keyword = ''.join(chr(x + 97) for x in word)
                    option = self.decrypt_vigenere(ciphertext, keyword)
                    prob = self.judge_plaintext(option)
                    options.append((option, keyword, prob))
                    
            options.sort(key=lambda x: x[2], reverse=True)
            return options   
        
    def brute_force_multicipher(self, ciphertext, possible_keywords=None):
        plaintext_options = []
        
        cipher_types = [
            self.brute_force,
            self.brute_force_caesar,
            self.brute_force_caesar_variant,
            self.brute_force_keyword,
            self.brute_force_vigenere,
        ]
        
        for f in cipher_types:
            plaintext_options.extend(f(ciphertext))
            
        plaintext_options.sort(key=lambda x: x[2], reverse=True)
        
        return plaintext_options   

if __name__=="__main__":
    cipher = Affine()
    print(cipher.brute_force_caesar('R UXEN VH TRCCH,'))