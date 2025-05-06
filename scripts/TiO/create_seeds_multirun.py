import random

num_seeds = 10
for _ in range(5):
    seeds = [str(random.randint(10000000, 99999999)) for _ in range(num_seeds)]
    string=str(seeds[0])
    for seed in seeds[1:]:
        string=string+','+str(seed)
    print(string) 
    with open('seeds.txt', 'a') as file:
        file.write(string + '\n')   
