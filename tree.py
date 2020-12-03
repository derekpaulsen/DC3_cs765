# "Tree of Stuff" Data Processing Code
# written November, 2019 by Michael Gleicher to prepare data for CS765
#
# This code reads the metadata files from the UCSD crawls of Amazon products
# and builds simpler representations of the tree of product categories
#
# This version was edited after 11/14/2019

# use standard data types
from typing import List
from collections import Counter, defaultdict

# use standard libraries
import csv
import time
import os
import gzip
import pickle

# this saves me from having to put the full pathname to the data file
# all the time
# it also opens the file with the gzip library if necessary
dataPath = "c:/users/gleicher/Downloads/Amazon Reviews"

def openFile(name):
    if os.path.exists(name):
        if name[-3:] == ".gz":
            return gzip.open(name,"rt")
        else:
            return open(name)
    elif os.path.exists(dataPath + "/" + name):
        return openFile(dataPath + "/" + name)
    else:
        raise FileNotFoundError("Can't find `{}'".format(name))


# simple class to store the tree
# note: this store some extra stuff for future use
# it's actually a bit of a pain to use an object rather than a dictionary
# since it can't be json dumped :-(
class Node:
    def __init__(self, name:str, id:int, path:List[str]):
        self.name = name
        self.id = id
        self.path = path
        self.exampleProduct = None
        self.parent = None
        self.children = dict()
        self.productCount = 0
        self.subtreeProductCount = 0
        self.also = Counter()
    def __repr__(self):
        return "<{}:{}:{}>".format(self.id,self.name,len(self.children))


def getNode(root:Node, nodes:List[Node], cat:List[str]):
    """
    find a node in the rooted tree - add nodes (and parents) if necessary

    :param root: root of the tree
    :param nodes: list of nodes (so we can add a new one if we make it)
    :param cat: category to add
    :return: the node for cat
    """
    ptr = root
    for i,cw in enumerate(cat):
        if cw in ptr.children:
            ptr = ptr.children[cw]
        else:
            newnode = Node(cw,len(nodes),cat[:i+1])
            nodes.append(newnode)
            newnode.parent = ptr
            ptr.children[cw] = newnode
            ptr = newnode
    return ptr


def buildTreeFromFile(filename="metadata.json.gz", max=10000, outname="test", alsos=True):
    """
    This is the real workhorse of data processing - takes in a metadata file and
    returns the tree. Note: it also creates a set of files. The files are named
    with "outname" as a prefix. So, you would get:

    test-nodes.csv (list of all the node information - the format we use for class)
    test-tree.pickle (a pickle file of the tree)

    :param filename:
    :param max:
    :param outname:
    :return:
    """
    start = time.time()
    root = Node("root",0,[])
    nodes = [ root ]
    root.id = 0
    nocats = 0
    prods = 0
    pairs = Counter()

    with openFile(filename) as fp:
        # this is basically, for each product
        for i, line in enumerate(fp):
            if max and i >= max:
                break
            if (i % 20000 == 0):
                print("Record {}".format(i))
            record = eval(line)
            # keep track of the set of node the product is UNDER, so we can count it once
            prodNodes = set()
            try:
                prods += 1
                cats = record["categories"]
                # process each category
                # count the product in each category
                for cat in cats:
                    node = getNode(root,nodes,cat)
                    if not(node.exampleProduct):
                        node.exampleProduct = record
                    node.productCount += 1

                    # add to the set of nodes this product is in
                    pn = node
                    while pn:
                        prodNodes.add(pn)
                        pn = pn.parent

                    # an n^2 process - what else are you on the list of?
                    if alsos and (len(cats) > 1):
                        for c2 in cats:
                            n2 = getNode(root,nodes,c2)
                            if n2 != node:
                                node.also[n2.id] += 1
                # now that we've processed all categories we can add this produce
                # to all of them
                for k in prodNodes:
                    k.subtreeProductCount += 1
            except KeyError:
                nocats += 1

    print("Products {}, NoCats {}, Nodes {}".format(prods, nocats, len(nodes)))

    aname = "" if alsos else "-noalsos"

    writeCSV(outname+aname+"-nodes.csv",nodes)

    with open(outname+aname+"-tree.pickle","wb") as fo:
        pickle.dump(root,fo)

    print("Ran for ",time.time()-start)

    return root

def writeCSV(fname, nodes):
    """
    Write the data for a list of nodes to a CSV file
    once was part of buildTreeFromNodes
    :param fname:
    :param nodes:
    :return:
    """
    with open(fname,"w",newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id","name","productCount","subtreeProductCount","parent","numChildren","pathName", "children", "alsoCount", "also"])
        # don't write the root
        for n in nodes:
            children = [n.children[c].id for c in n.children]
            alsos = n.also.most_common(200)
            w.writerow([n.id,n.name,n.productCount,n.subtreeProductCount,n.parent.id if n.parent else 0,len(n.children),n.path,children,len(n.also),alsos])

def getNodeList(tree):
    """
    reconstruct the node list from a tree
    :param tree:
    :return:
    """
    nodes = [ ]

    def addToNodes(node):
        nodes.append(node)
        for c in node.children:
            addToNodes(node.children[c])
    addToNodes(tree)
    nodes.sort(key=lambda x:x.id)
    return nodes

def subtree(tree,category="Pet Supplies"):
    """
    pull out a subtree from a bigger tree
    category must be a name of a direct child (top level branch)
    this tries to do the right thing about renumbering and avoiding "alsos" outside of the tree
    :param tree:
    :param category:
    :return:
    """
    if category not in tree.children:
        raise KeyError("Category Not in Root")
    oldroot = tree.children[category]
    oldnodes = getNodeList(tree)
    oldToNew = dict()

    root = Node(category,0,[category])
    nodes = [root]
    # temprarily store the corresponding node as the product count
    # this gets changed when the node is processed
    root.productCount = oldroot

    queue = [root]

    while queue:
        newnode = queue.pop()
        oldnode = newnode.productCount
        oldToNew[oldnode.id] = newnode
        newnode.productCount = oldnode.productCount
        for c in oldnode.children:
            oldchild = oldnode.children[c]
            newchild = Node(c,len(nodes),newnode.path + [c])
            newchild.productCount = oldchild
            newchild.parent = newnode
            newchild.subtreeProductCount = oldchild.subtreeProductCount
            newchild.also = oldchild.also
            newchild.exampleProduct = oldchild.exampleProduct
            nodes.append(newchild)
            newnode.children[c] = newchild
            queue.append(newchild)

    # now make another pass to clean up the alsos
    for n in nodes:
        oldAlso = n.also
        n.also = Counter()
        for a in oldAlso:
            try:
                newNode = oldToNew[a]
                n.also[newNode.id] = oldAlso[a]
            except KeyError:
                pass

    return root

def writeSubtree(tree, category, filename):
    t = subtree(tree,category)
    writeCSV(filename,getNodeList(t))