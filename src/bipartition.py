class Node:

    def __init__(self, x, n):

        self.childs = []

        self.data = x

        self.lvl = n
        
    def __repr__(self):
        return self.data

    def addToChilds(self, x):

        nodex = Node(x, n+1)
        self.childs.append(nodex)



    

def addToList(t,a,b):
    if t == None:
        return
    else:
        if t.lvl % 2 == 0:
            if(len(a) > t.lvl//2):
                print( "adding to a lvl: " + str(t.lvl) + " added: " + t.data  )
                a[t.lvl//2].append(t)
                for i in range(len(t.childs)):
                    addToList(t.childs[i], a, b)
            else:
                
                newlvl = []
                newlvl.append(t)
                print( "adding to a lvl: " + str(t.lvl) + " added: " + t.data  )
                a.append(newlvl)
                for i in range(len(t.childs)):
                    addToList(t.childs[i], a, b)

        else:
            if(len(b) > t.lvl//2):
                print( "adding to b lvl: " + str(t.lvl) + " added: " + t.data  )
                b[t.lvl//2].append(t)
                for i in range(len(t.childs)):
                    addToList(t.childs[i], a, b)
            else:
                newlvl = []
                newlvl.append(t)
                print( "adding to b lvl: " + str(t.lvl) + " added: " + t.data  )
                b.append(newlvl)
                for i in range(len(t.childs)):
                    addToList(t.childs[i], a, b)
                
                
def bipartition(t):
    a = []
    b = []
    addToList(t,a,b)
    #node lvl even sa a ya lvl listine ekle
    #node lvl odd sa b ya lvl listine ekle
    bipart = []
    bipart.append(a)
    bipart.append(b)

    return bipart
                
t = Node('a', 0)
t.childs.append(Node('b',1))
t.childs[0].childs.append(Node('c',2))
t.childs[0].childs.append(Node('d',2))
t.childs[0].childs.append(Node('e',2))
t.childs[0].childs[0].childs.append(Node('f',3))
t.childs[0].childs[1].childs.append(Node('g',3))
t.childs[0].childs[1].childs.append(Node('z',3))
t.childs[0].childs[1].childs.append(Node('j',3))
t.childs[0].childs[2].childs.append(Node('l',3))
t.childs[0].childs[0].childs[0].childs.append(Node('i',4))
t.childs[0].childs[0].childs[0].childs.append(Node('g',4))
t.childs[0].childs[0].childs[0].childs.append(Node('h',4))

retval = bipartition(t)


print(retval[0])
print()
print(retval[1])






