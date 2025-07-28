import os

with open('merge.txt', 'w', encoding='utf-8') as outfile:
    for filename in os.listdir('.'):
        if filename.endswith('.txt') and filename != 'merge.txt':
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read().rstrip('\n') + '\n')
