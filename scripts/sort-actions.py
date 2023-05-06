import sys

if __name__ == '__main__':
  file = sys.argv[1]
  offset = int(sys.argv[2])
  actions = {}
  src_lengths = {}
  hyp_lengths = {}
  with open(file, 'r') as fd:
     for _ in range(offset):
       fd.readline()
     for line in fd:
       if line.startswith('|'):
         continue
       elif line.startswith('S-'):
         id = line.split('\t')[0].split('-')[1]
         src_lengths[id] = line.split('\t')[1].split()
       elif line.startswith('T-'): 
         continue
       elif line.startswith('H-'):
         id = line.split('\t')[0].split('-')[1]
         hyp_lengths[id] = line.split('\t')[2].split()
       elif line.startswith('A-'):
         id = line.split('\t')[0].split('-')[1]
         actions[id] = line.split('\t')[2]

  for i in range(len(actions)):
    try:
      if len(src_lengths[str(i)]) >= actions[str(i)].count('4'):
        sys.stdout.write(str(actions[str(i)][:-1] ) + '\n')
      else:
        sp_actions = actions[str(i)].split()
        last_read_idx = ''.join(sp_actions).rindex('4')
        del sp_actions[last_read_idx]        
        sys.stdout.write(str(actions[str(i)][:-1] ) + '\n')
    except KeyError:
      print("Error in", i)
      continue
