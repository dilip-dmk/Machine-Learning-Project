filename = '3.txt'

num = 0

with open(filename, 'r') as f:
  fw = open('3c.txt', 'w')
  while True:
    newline = []
    line = f.readline()
    if not line:
      break
    print("processing line #" + str(num))
    num = num + 1
    arr = line.split(' ')
    for item in arr:
        word = item.split('/')
        newline.append(word[0])

    l =  "".join(newline)
    fw.write(l)
    print l,
  fw.close()
print("done")
    
