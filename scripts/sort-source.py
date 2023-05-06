import sys

if __name__ == '__main__':
  file = sys.argv[1]
  offset = int(sys.argv[2])
  sents = {}
  with open(file, 'r') as fd:
     for _ in range(offset):
       fd.readline()
     for line in fd:
       if line.startswith('|'):
         continue
       elif line.startswith('T-') or line.startswith('A-'):
         continue
       elif line.startswith('S-'):
         id = line.split('\t')[0].split('-')[1]
         if id not in sents:
             sents[id] = [line.split('\t')[1]]
         else:
             sents[id].append(line.split('\t')[1])

  for i in range(len(sents)):
    if str(i) in sents:
      for elem in sents[str(i)]:
        sys.stdout.write(elem)
