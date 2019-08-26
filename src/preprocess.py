import os
cwd = os.getcwd()[:-4]
# ========================== Class Definitions ==============================
class Tree:
    def __init__(self,data=None):
        self.root = data
        
    def pr(self):
        self.root.pr()
    def __repr__(self):
        return self.root.__repr__()
        
    def set_root(self, data):
        self.root = data
    
    def get_root(self):
        return self.root
    
    def addPath(self,ll):
        if self.root == None:
            #print(ll[0], 'created')
            self.root = Nodes(ll[0],[])
        #print('will add', ll[1:])
        self.root.addPath(ll[1:])
        
class Nodes:
    def __init__(self,val = None, childs = []):
        self.val = val
        self.childs = childs
        
        
    def pr(self):
        print('my value:',self.val)
        print('')
        #print('my childs:', self.childs)
        for i in range(len(self.childs)):
            self.childs[i].pr()
            
     # prints data
    def __repr__(self):
        
        #for i in self.childs:
            #print(i)
        return str(self.val)
    
   
            
    # returns the index of given data in childs else -1
    def isInChild(self, x):
        
        for i in range(len(self.childs)):
            if self.childs[i].val == x:
                
                return i
               
        return -1
            
    # if x data is not in childs adds it to the child        
    def addToChilds(self, x):
        #print('addtochild called on', self.val, self.childs)
        index = self.isInChild(x)
        #print(x, 'exist @', index)
        if index == -1:
            #print(x, 'added to child')
            nodex = Nodes(x,[])
            self.childs.append(nodex)
            return nodex
        else:
            return self.childs[index]
        
    # gets a path as a list adds to the Tree    
    def addPath(self, path):
        #print('addpath called on', self.val)
        if len(path) == 0:
            return
        else:
            self.addToChilds(path[0])
            index = self.isInChild(path[0])
            #print('confirm index added @', index, self.childs)
            dummy = self.childs[index]
            #print('calling addpath', path[1:], 'on', dummy)
            dummy.addPath(path[1:])
            return


        
        
class NodeLabels():
    def __init__(self):
        self.pairs = {}
        self.reversepairs = {}
        self.cur_label = 0
    def addNode(self,val):
        if not val in self.pairs.keys():
            self.reversepairs.update({str(self.cur_label) : str(val)})
            self.pairs.update({val : str(self.cur_label)})
            self.cur_label = self.cur_label + 1
            
    def getVal(self, key):
        return int(self.pairs[key])
    
    def get(self):
        return self.pairs
    
    def get_reverse(self):
        return self.reversepairs

# ========================== Helpers Functions ===================================

def findrootkey(rough_tree):
    cur_root = None
    for i in rough_tree.keys():
        if not keyinvalues(i,rough_tree.values()):
            cur_root = i
            return cur_root
    return cur_root
        
        
def keyinvalues(key, ll):
    for i in ll:
        if key in i:
            return True
    return False

def reconstNode(rough_tree, key):
    childs = []
    if not key in rough_tree.keys():
        return Nodes(key,childs)
    
    c = rough_tree[key]
    for i in c:
        childs.append(reconstNode(rough_tree, i))
    
    return Nodes(key, childs)
            
  
def accedges(node, parent, edges, dic):
    if parent == None:
        for i in range(len(node.childs)):
            accedges(node.childs[i],node.val, edges, dic)
    else:
        dic.addNode(parent)
        dic.addNode(node.val)
        edges.append([dic.getVal(parent), dic.getVal(node.val)])
        for i in range(len(node.childs)):
            accedges(node.childs[i],node.val, edges, dic)
    return {'edges': edges, 'labels':dic.get_reverse(), 'r_labels':dic.get()}

def jsontoTree(json_dict):
    edges = json_dict['edges']
    labels = json_dict['r_labels']
    edges_dict = {}
    for i in edges:
        if labels[str(i[0])] in edges_dict.keys():
            edges_dict[labels[str(i[0])]].append(labels[str(i[1])])
        else:
            edges_dict.update({labels[str(i[0])] : [labels[str(i[1])]]})
    return edges_dict

def subtrees_node(node, st):
    if node == None:
        return
    if node.childs == []:
        return
    for i in node.childs:
        subtrees_node(i, st)
    st.append(Tree(node))
    
# Creates an array of objects
def pathTolistform(path):
          
    return path.split('/')

# Dot product
def completePaths(path, lof):
    comp = []
    for i in lof:
        name = i
        if name == None:
            continue
        cp = path + '/' + name
        
        comp.append(pathTolistform(cp))
   
    return comp

# Crawls the given path and extracts the omit path
def crawl(svn_root_path, omit=""):
    print("CRAWLING STARTED")
    print('')
    import os
    from os.path import join, getsize
    paths = []
    txtfiles = []
    for root, dirs, files in os.walk(svn_root_path):
        if files != []:
            paths.append(root.strip(omit))
            txtfiles.append(files)
    print("CRAWLING ENDEDD")
    print('')
    #print("Number of paths found: ",len(paths))
    print('')
    return [paths,txtfiles]

# Merges two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# Dot product on all elements
def completeallPaths(paths, llof):
    
    all_l = []
    del_c ={}  
    for i in range(len(paths)):
        partial = completePaths(paths[i], llof[i])
        cur_key = partial[0][1]
        if cur_key in del_c.keys():
            del_c[cur_key] = del_c[cur_key] + len(partial)
        else:
            del_c.update({cur_key : len(partial)})
        all_l = all_l + partial
    print('found',del_c)
    print(' ')
    return all_l

# Adds paths to a tree
def lltoTree(path, retval = Tree(None)):
    for i in range(len(path)):
        #print(path[i])
        retval.addPath(path[i])
    return retval

# Given a list of root trees from the root directory extracts the elements as trees
def getAllTrees(lrootTree):
    allTrees = []
    for rootTree in lrootTree:
        for i in rootTree.root.childs:
            allTrees.append(Tree(i))
    return allTrees

# Gets trees generates json format data
# If test_train true will generate labeling as weel for testing and training
def getAlledges(allTrees, test_train = True):
    alledges = []
    target_dic = {}
    reverse_target_dic ={}
    cur_target = 0
    for i in allTrees:
        edges = getedges(i)
        if test_train:
            target = i.root.val
            if not target in reverse_target_dic.keys():
                reverse_target_dic.update({target:cur_target})
                target_dic.update({cur_target:target})
                cur_target = cur_target + 1
            edges.update({'target':reverse_target_dic[target]})
        alledges.append(edges)
    return alledges, target_dic

# Deletes 5% to 20% of given list randomly
def deleteRandomPaths(allPaths):
    import random
    mylist = allPaths
    n_elements = random.randint(len(mylist)*5//100, len(mylist)*20//100)
    print('deleting',n_elements,'out of',len(mylist))
    print(' ')
    random.shuffle(mylist)
    del_c ={}
    for i in range(0,n_elements):
        cur_key = mylist[0][1]
        if cur_key in del_c.keys():
            del_c[cur_key] = del_c[cur_key] + 1
        else:
            del_c.update({cur_key : 0})
        mylist.pop
        random.shuffle(mylist)
    print('deleted:', del_c)
    print(' ')
    return mylist  

# ==================== Tree JSON cnversion / subgraph ===================
# Returns the subtrees of a given tree
def subtrees(tree):
    st = []
    subtrees_node(tree.root, st)
    return st
    
# Gets a tree returns it as dictionary
def getedges(t):
    edges = []
    curNode = t.root
    dic = NodeLabels()
    return accedges(curNode, None, edges, dic)

# Gets a dictinary construct the tree
def recontructTree(json_dict):
    rough_tree = jsontoTree(json_dict)
    cur_tree = Tree()
    root_key = findrootkey(rough_tree)
    print(root_key)
    cur_tree.set_root(reconstNode(rough_tree,root_key))
    return cur_tree

# Takes the root directory of samples, and unneccessary preceding apth
# If test_train true class mappings and labeling will be added
# If createRandomData True randomsamples(100) new random trees will be generated
def createData(path = cwd + '/input/test_oss/', omitpath = cwd + '/input/', test_train = True, createRandomData = True, Randomsamples = 100, realsamples = 1):
    import copy
    crawl_Res = crawl(path, omitpath)
    allPaths = completeallPaths(crawl_Res[0],crawl_Res[1]) # remove random paths to create test data
    rootTree = lltoTree(allPaths) # one tree
    lrootTree = []
    lrootTree.append(rootTree)
    for i in range(realsamples-1):
        print('creating deep copy of trees')
        print(' ')
        lrootTree.append(copy.deepcopy(rootTree))
    if createRandomData:
        print('creating randomized trees')
        print(' ')
        for i in range(Randomsamples):
            lrootTree.append(lltoTree(deleteRandomPaths(allPaths)))
        
        
    allTrees = getAllTrees(lrootTree) #list of trees
    
    allJSON = getAlledges(allTrees, test_train) #list of json, class mapping
    print('Class Mappings: ', allJSON[1])
    print(' ')
    print('Number of samples: ', len(allJSON[0]))
    return allJSON

def saveAsJSON(scan_path = cwd + '/input/test_oss/', scan_omit_path = cwd + '/input/', test_split = 2, train_split = 8, save_path = cwd + '/input/', test_train = True, createRandomData = True, Randomsamples = 0, realsamples = 1):
    raw_dic_data, map_data =  createData(scan_path, scan_omit_path, test_train, createRandomData, Randomsamples,realsamples)
    import json
    data = json.dumps(map_data)
    f = open(save_path + "map_data.json","w")
    f.write(data)
    f.close()
    order = 0
    train_number = train_split * len(raw_dic_data) // (test_split + train_split)
    #print(train_number)
    for i in raw_dic_data:
        #print(str(order)+'.json',i['target'])
        data = json.dumps(i)
        f = open(save_path + 'test/'+ str(order) + ".json","w")
        f.write(data)
        f.close()
        f = open(save_path + 'train/'+ str(order) + ".json","w")
        f.write(data)
        f.close()
        order = order + 1

saveAsJSON(Randomsamples = 0, realsamples = 100)        