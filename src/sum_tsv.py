import csv

# read b,m,s,d cate .tsv
bmcate = open('bmcate.tsv', 'r')
bm_r = list(csv.reader(bmcate, delimiter='\t'))

scate = open('scate.tsv', 'r')
s_r = list(csv.reader(scate, delimiter='\t'))

dcate = open('dcate.tsv', 'r')
d_r = list(csv.reader(dcate, delimiter='\t'))

# write cate output.tsv
cate = open('output.tsv', 'w')
wr = csv.writer(cate, delimiter='\t')

for i in range(len(list(bm_r))):
  wr.writerow([bm_r[i][0], bm_r[i][1], bm_r[i][2], s_r[i][1], d_r[i][1]])

bmcate.close()
scate.close()
dcate.close()
cate.close()