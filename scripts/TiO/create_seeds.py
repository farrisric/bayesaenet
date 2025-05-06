import random

# Define the number of seeds to generate
num_seeds = 5  # You can change this number as needed

# Generate the seeds
seeds = [str(random.randint(10000000, 99999999)) for _ in range(num_seeds)]

# Save the seeds to a file
with open('seeds.txt', 'w') as file:
    for seed in seeds:
        file.write(seed + '\n')

print(f'{num_seeds} seeds have been generated and saved to seeds.txt')
