i=open('.../goldstandard_trinary.txt')
r=i.read()
i.close()
r=r.split('\n')
d=[]
r.pop(0)
for e in r:
	e=e.strip()
	d.append(e.split('\t'))

name={1:'manual',2:'svm',3:'lexi',4:'huliu',5:'geninq',6:'LMD',7:'roBERTa_senti'}
v=['-1','0','1']

for a in range(7):
	for b in range(7):
		if a<b:
			for v1 in v:
				for v2 in v:
					c=0
					for e in d:
						if e[a+1]==v1 and e[b+1]==v2:
							c=c+1
					print(name[a+1],name[b+1],v1, v2, c)
					
