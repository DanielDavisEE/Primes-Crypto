import math, itertools, re, random
"""
alphabet_length = 10

cipher = '00001111222233344455667'
#cipher = '877666555544444333333222222211111111000000000'

frequency_list=None
if not frequency_list:
    frequency_list = list('2890437651')

letter_counts = [0] * alphabet_length
for letter in cipher:
    index = ord(letter) - 48
    letter_counts[index] += 1

# Produce an alphabet sorted by frequency in cipher and then in order from frequency_list
letter_rank = list('0123456789')
letter_rank.sort(key=lambda x: frequency_list.index(x))
letter_rank.sort(key=lambda x: letter_counts[ord(x) - 48], reverse=True)


# A dict with keys as number of occurrances and values as the letters
count_dict = {}
for i, count in enumerate(letter_counts):
    count_dict[count] = count_dict.get(count, []) + [chr(i+48)]

# A list which gives the number of letters which have the same number of
#    occurrances as each letter
common_occurrence = [0] * alphabet_length
for occurrence, letters in count_dict.items():
    for letter in letters:
        common_occurrence[ord(letter)-48] = len(letters)

# A list which groups together numbers with equal number of occurrances and assigns a number     
occurrance_group = [0] * alphabet_length
occurrance_dict = {}
group = 0
current_group = letter_counts[ord(letter_rank[0])-48]
for l in letter_rank:
    if letter_counts[ord(l)-48] != current_group:
        current_group = letter_counts[ord(l)-48]
        group += 1
        
    occurrance_group[ord(l)-48] = group
    occurrance_dict[group] = occurrance_dict.get(group, []) + [l]

num_groups = group + 1

# A list which gives the number of letters prior to the group of this letter,
#    grouped by number of occurrances
previous_letters = [0] * alphabet_length
count = 0
occurrence = common_occurrence[ord(letter_rank[0])-48]
tmp = occurrence
for i in range(len(previous_letters)):
    if tmp == 0:
        count += occurrence
        occurrence = common_occurrence[ord(letter_rank[i])-48]
        tmp = occurrence
        
    previous_letters[ord(letter_rank[i])-48] = count
    
    tmp -= 1
    
options = 1
for n in set(letter_counts):
    x = len(count_dict[n])
    options *= math.factorial(x)


for i in range(alphabet_length):
    l = ord(letter_rank[i]) - 48
    n = letter_counts[l]
    m = previous_letters[l]
    print(f'cipher letter = {letter_rank[i]}, frequency order = {frequency_list[i]}, occurrance = {n}, {m}') 

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
        if letter_counts[ord(occurrance_dict[group][0])-48] == 0:
            iterator_template[group] = constant_iter, num_options[group]
            
    print(iterator_template)
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
        i = ord(letter) - 48
        previous = previous_letters[i]
        group = occurrance_group[i]
        
        group_index = l - previous

        permutation = c[group][group_index]
        
        decrypt_dict[letter] = frequency_list[previous + permutation]
    decrypt_dict_options.append(decrypt_dict) 

"""

#for i in range(options):
    #decrypt_dict = {}
    
    
    
    
    #min_index = 0
    #for l in range(alphabet_length):
        #i = ord(letter_rank[l])-48
        
        #m = common_occurrence[i]
        
        #p = group_iterations[m][0][group_iterations[m][1]]
        #previous = previous_letters[i]
        #group_index = l - previous
        
        #decrypt_dict[letter_rank[l]] = frequency_list[previous + p[group_index]]
        
        ##print(p, '\t', letter_rank[l], previous + p[group_index], previous, group_index, p[group_index])
        
        #if group_index == m - 1:
            #group_iterations[m][1] += 1
            #if group_iterations[m][1] == len(group_iterations[m][0]):
                #group_iterations[m][1] = 0
        
    #decrypt_dict_options.append(decrypt_dict)
    
#tmp_set = set()
#for d in decrypt_dict_options:
    
    #tmp_set.add(''.join(d.values()))
    
    #print(*d.keys())
    #print(*d.values())
    #print()

#text = """They
#were admirable things for the observer--excellent for drawing the
#veil from men's motives and actions."""


#delimiters = [",", ";", ".", ":", " ", "(", ")", "/", "\n", "-"]
#regexPattern = "('s)?[" + ''.join(map(re.escape, delimiters)) + ']+'
#print(regexPattern)
#word_list = [word for word in re.split(r'[^a-zA-Z0-9\']+', text)][:-1]

#print(word_list)

iterations = 10000000
roll_3d6 = 0
roll_3d6_reroll = 0
roll_4d6_drop = 0

for _ in range(iterations):
    rolls = [random.randint(1, 6), random.randint(1, 6), random.randint(1, 6)]
    new_roll = random.randint(1, 6)
    rolls.sort()
    roll_3d6 += sum(rolls)
    roll_3d6_reroll += sum(rolls[1:]) + new_roll
    roll_4d6_drop += sum(rolls) if new_roll < rolls[0] else sum(rolls[1:]) + new_roll

roll_3d6  /= iterations
roll_3d6_reroll  /= iterations
roll_4d6_drop  /= iterations

print(f'Roll 3d6: {roll_3d6:.3f}')
print(f'Roll 3d6 reroll lowest: {roll_3d6_reroll:.3f}')
print(f'Roll 4d6 drop lowest: {roll_4d6_drop:.3f}')
