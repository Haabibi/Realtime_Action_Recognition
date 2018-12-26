txt_path = './result_hmdb_flow_multi.txt'
with open(txt_path) as f:
    lines = f.readlines()
    result = {}
    avg_time = 0
    for line in lines[:-1]:
        foo = line.split('\t')
        if foo[1] not in result.keys():
            result.update({foo[1]: [foo[0]]})
        else: 
	    result[foo[1]].append(foo[0]) 
           # print(type(result[foo[1]]))
        one_time = float(foo[2][:6])
        avg_time += one_time 
    avg_time = avg_time / (len(lines)-1)

#print(result.keys())

up_result = {}
for key in result.keys():
   # print(key, result[key])
    li = result[key]
#    num_a, num_f = 0, 0 
#    for i in li:
#        if i[0] == 'A':
#            num_a += 1 
#        if i[0] == 'F':
#            num_f += 1
    foo = len(li)
    up_result.update({key: len(li)})

print(up_result)
'''
temp_dict = {'Fall': 0, 'ADL': 0}
for key in result.keys():
    value_li = result[key]
    for item in value_li:
        if item[0] == 'A':
            temp_dict['ADL'] += 1 
        else:
            temp_dict['Fall'] += 1 
    print(temp_dict)  
    print(value_li)

print(avg_time)
print(result)
print(result.keys())
print(len(result.keys()))
print(len(lines)-1)
'''
