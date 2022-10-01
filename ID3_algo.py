import numpy as np
import sys
import pprint

eDic = {}
columns = []
def entropy(v1,v2):
    if (v1 == v2):
        return 1.0
    elif (v1 == 0  or v2 ==0 ):
        return 0.0
    p1 = float(v1)/float(v1+v2)
    p2 = float(v2)/float(v1+v2)
    return (-(p1 * np.log2(p1)) - (p2*np.log2(p2)))

# A general function which gives the root node from the table passed to it
def selectRoot(table,encodings):
    table_length = len(table)
    info_gain = []
    #convert encodings to int approporiately
    ec = []
    for i in range(len(columns)):
        if columns[i] in encodings:
            ec.append(i)

    for i in ec:
        c_yes = np.count_nonzero(table[:,-1])
        c_no = table_length - c_yes
        entropy_before = entropy(c_yes,c_no)
    
        #find out yes and no instances of each values
        values, counts = np.unique(table[:,i],return_counts=True) #For finding out unique values inside an attribute
        entropy_after = 0
        for j in range( len(values)):
            prob = counts[j] / table_length
            labels  = np.where(table[:,i] == values[j] )
            subtable = table[labels]
            yes_instances = np.count_nonzero(subtable[:,-1])
            no_instances = len(subtable) - yes_instances
            entropy_after = entropy_after + prob * entropy(yes_instances,no_instances)
        info_gain_single = entropy_before - entropy_after 
        info_gain.append(info_gain_single)
    index = np.argmax(np.array(info_gain))
    root = encodings[index]
    return root



# A helper function which takes in the initial dataset and repeatdely finds the root node
def Tree(initial_dataset, formatting_rules):
    table_length = len(initial_dataset)
    yes_instances = np.count_nonzero(initial_dataset[:,-1])

    if (yes_instances == table_length):
        return "Yes"
    elif (yes_instances == 0):
        return "No"
    else:
        t1 = {}
        root  = selectRoot(initial_dataset,formatting_rules)
        col = 0
        for i in range( len(columns) ):
            if root == columns[i]:
                col = i
                break
        values, _ = np.unique(initial_dataset[:,col],return_counts=True)
        for i in range(len(values)): 
            #filter table
            labels  = np.where(initial_dataset[:,col] == values[i] )
            subtable = initial_dataset[labels]
            att_name = eDic[root][values[i]]
            try:
                formatting_rules.remove(root)
            except:
                pass
            t1[att_name] = Tree(subtable,formatting_rules)#formatting_rules)
    
        return { root:t1 }

def encodeDict(filename):
    global eDic 
    global columns
    with open(filename, "r") as file: 
        data = file.readlines()
        columns = data[0].split(",")
    

        for i in range( len(data[0].split(',')) ):
            values = data[i+1].split(',')
            miniDic = {}
            for j in range( len( data[i+1].split(',') ) ):
                v = values[j].rstrip("\n")
                miniDic[j] = v
            
            k = data[0].split(',')
            k = k[i].rstrip("\n")
            eDic[ k ] = miniDic
    return eDic


def main():
    dataset = sys.argv[1]
    encodings = sys.argv[2]
    array=np.loadtxt(dataset,dtype=int, delimiter=",")
    eDic = encodeDict(encodings)
    tree = Tree(array,list(eDic.keys()))
    pprint.pprint(tree)







if __name__ == "__main__":
    main()
