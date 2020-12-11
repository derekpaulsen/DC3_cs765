import pickle


class CompactNode:
    __slots__ = [
        'name',
        'id',
        'path',
        'parent',
        'children',
        'productCount',
        'subtreeProductCount',
        'also',
    ]

    def __init__(self, n):
        self.name = n.name
        self.id = n.id
        self.path = n.path
        self.parent = n.parent
        self.children = n.children if len(n.children) else None
        self.productCount = n.productCount
        self.subtreeProductCount = n.subtreeProductCount
        self.also = n.also

    def __repr__(self):
        return "<{}:{}:{}>".format(self.id,self.name,len(self.children))


def convert(t):
    for k in t.children:
        t.children[k] = convert(t.children[k])

    return CompactNode(t)




def main():
    with open('tree-all.pickle', 'rb') as ifs:
        t = pickle.load(ifs)

    ct = convert(t)
    with open('compact_tree.pickle', 'wb') as ofs:
        pickle.dump(ct, ofs)


if __name__ == '__main__':
    main()



