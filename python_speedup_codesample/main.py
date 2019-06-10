attributes = ['name', 'dob', 'gender']
values = [['jason', '2000-01-01', 'male'],
['mike', '1999-01-01', 'male'],
['nancy', '2001-02-01', 'female']
]

dc={}
rs=[{attributes[y]:x[y]  for y in range(len(attributes))}  for x in values]

rs3=[dict(zip(attributes,v)) for v in values]

print(rs3)

print("z;",dict(zip(attributes,values[0])))
rs2=[]

for i in range(len(values)):
    dc2={}
    for j in range(len(attributes)):
        dc2[attributes[j]]=values[i][j]
    rs2.append(dc2)
#print(rs2)