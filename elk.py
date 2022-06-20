f = open('log.txt', 'r')
q = open('audio_log.txt', 'r')

l, r, u, d = 0, 0, 0, 0

p, n, m = 0, 0, 0

a = 0

for line in f.readlines():
    if "LEFT" in line:
        l += 1
    elif "RIGHT" in line:
        r += 1
    elif "UP" in line:
        u += 1
    elif "DOWN" in line:
        d += 1
    elif "Mobile" in line:
        p += 1
    elif "No Person" in line:
        n += 1
    elif "Many" in line:
        m += 1

for line in q.readlines():        
    if "Audio" in line:
        a += 1

print("Head Position : UP --> ", u)
print("Head Position : DOWN --> ", d)
print("Head Position : RIGHT --> ", r)
print("Head Position : LEFT --> ", l)
print("YOLO : Mobile Phones --> ", p)
print("YOLO : No Person --> ", n)
print("YOLO : Many Persons --> ", m)
print("AUDIO : Noise --> ", a)

# w1 = 0/7 
# w2 = 0/7
# w3 = 1/7
# w4 = 2/7
# w5 = 3/7
# w6 = 5/7
# w7 = 6/7

S = (1/7)*l + (1/7)*r + (1/7)*u + (2/7)*d + p + (3/7)*n + (5/7)*m + (6/7)*a

print("Plagiarism Score :",S)