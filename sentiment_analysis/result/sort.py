filename = 'bbresult'
num = 1
result = []

pos = 0
mid = 0
neg = 0

with open(filename, 'r') as f:
  while True:
    line = f.readline()
    if not line:
      break
    print("processing line #" + str(num))
    num = num + 1
    arr = line.split(' ')
    if len(arr) < 3:
      continue
    print arr[1], arr[2], "".join(arr[3:])
    if arr[1] == 'pos':
      pos = pos + 1
    if arr[1] == 'mid':
      mid = mid + 1
    if arr[1] == 'neg':
      neg = neg + 1
    item = (arr[1], float(arr[2]), "".join(arr[3:]))
    result.append(item)
  result.sort(key=lambda tup:(tup[0], tup[1]))
  print(len(result))
  f = open('sorted.log', 'w')
  for item in result:
    f.write(str(item[0]))
    f.write(",")
    f.write(str(item[1]))
    f.write(",")
    f.write(item[2])
  f.write("pos:%d\n" % pos)
  f.write("mid:%d\n" % mid)
  f.write("neg:%d\n" % neg)

print("done")
    
