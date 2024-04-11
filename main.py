s=list(set(input()))
arr=[]
for i in s:
    if i.isalpha():
        arr.append(i)
print(len(set(arr)))