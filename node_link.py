import plotly.graph_objects as go
import pickle
from pprint import pformat
import json
import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import logging
import logging.config
import sys
from tree import Node

LOGGING_CONF = './conf/logging.conf'

logging.config.fileConfig(LOGGING_CONF)

# initialize the logger
logger = logging.getLogger(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


depth = 0

# extract all the node labels
def get_nodes(node, labels):
    labels[node.id] = node
    for c in node.children.values():
        get_nodes(c, labels)


def add_links(node, source, target, value):
    for c in node.children.values():
        source.append(node.id)
        target.append(c.id)
        #value.append(c.subtreeProductCount)
        value.append(1)


class RenderNode:
    node_point_defaults = {
            'selected' : False
    }
    node_point_keys = [
            'label', 
            'id', 
            'parent_id',
            'parent_label',
            'depth',
            'node',
            'selected',
            'render_order'
    ]

    def __init__(self, node, parent):
        self._node = node
        self._parent = parent

        self._schildren = []
        #self._schild_ids = set()

    @property
    def id(self):
        return self._node.id

    @property
    def name(self):
        return self._node.name
    
    def __eq__(self, o):
        return self.id == o.id
    
    
    def add_child(self, node):
        rn = RenderNode(node, self)
        self._schildren.append(rn)
        self._schildren.sort(key=lambda x: x.name)
        #self._schild_ids.add(node.id)

    def remove_child(self, node):
        #self._schild_ids.remove(node.id)
        if node in self._schildren:
            self._schildren.remove(node)

    
    def toggle_node(self, id):
        # root node
        if self.id == id:
            self._schildren = []
            return True

        # remove node if already selected
        for c in self._schildren:
            if c.id == id:
                self.remove_child(c)
                return True

        # add node if node is not selected and child
        for c in self._node.children.values():
            if c.id == id:
                self.add_child(c)
                return True

        # continue search existing selected nodes
        for c in self._schildren:
            if c.add_node(id):
                return True


        # not in this subtree, continue search
        return False
            
                
    
    def make_node_point(self, **kwargs):
        t = tuple(kwargs.get(k, self.node_point_defaults.get(k)) for k in self.node_point_keys)

        if any(x is None for x in t):
            missing = [self.node_point_keys[i] for i in range(len(t)) if t[i] is None]
            raise RuntimeError(f'missing keys from call to make_node_point : {missing}')
        return t


    # render tree in dfs
    
    def render(self):
        if self._parent is not None:
            raise RuntimeError('render called on none root node')

        points = {}
        render_order = [0]
        self._render(points, 0, render_order)

        df = pd.DataFrame(points.values(), columns=self.node_point_keys)

        return df


    def _render(self, points, depth, render_order):
        points[self.id] = self.make_node_point(
                    label=self.name,
                    id=self.id,
                    depth=depth,
                    parent_id=self._parent.id if self._parent else np.nan,
                    parent_label= self._parent.name if self._parent else '',
                    node=self._node,
                    selected=True if self._parent else False,
                    render_order=tuple(render_order[:depth])
            )
        for c in self._node.children.values():
            points[c.id] = (self.make_node_point(
                    label=c.name,
                    id=c.id,
                    depth=depth+1,
                    parent_id=self.id,
                    parent_label=self.name,
                    node=c,
                    render_order=tuple(render_order)
            ))

        render_order = render_order + [0]
        for c in self._schildren:
            c._render(points, depth+1, render_order)
            render_order[-1] += 1
        




    
class Tree:

    def __init__(self, tree):
        self._tree = tree
        self._nodes = self._get_nodes()
        self.labels = self._get_labels(self._nodes)
        self._pos_to_node_id = [0]
        self._link_res = 30

        self._cos_y = np.cos(np.linspace(0, np.pi, self._link_res, endpoint=True))
        

        # add root node
        self._render_tree = RenderNode(self._tree, None)
        self._node_df = self._render_tree.render()
    
    def _get_nodes(self):
        nodes = {}
        get_nodes(self._tree, nodes)
        return nodes

    def _get_labels(self, nodes):
        label = [nodes.get(i) for i in range(0, max(nodes.keys()) + 1)]
        label = [(n.name if n else None) for n in label] 

        return np.array(label)


    def generate_layout(self):
        return {
            'height' : 2000,
            'showlegend' : False
        }

    def link_points(self, p1, p2):
        y = (self._cos_y  * (abs(p1[1] - p2[1]) / 2)) + ((p1[1] + p2[1]) / 2)
        if p2[1] > p1[1]:
            y = np.flip(y)

        x = np.linspace(p1[0], p2[0], len(y))

        return go.Scatter(
                    x = x,
                    y = y,
                    mode='lines',
                    name=None
                )
#        return go.Scatter(
#                    x = [p1[0], p2[0]],
#                    y = [p1[1], p2[1]],
#                    mode='lines'
#                )
#

    
    def make_node_point(self, **kwargs):
        t = tuple(kwargs.get(k, self._node_point_defaults.get(k)) for k in self._node_point_keys)

        if any(x is None for x in t):
            missing = [self._node_point_keys[i] for i in range(len(t)) if t[i] is None]
            raise RuntimeError(f'missing keys from call to make_node_point : {missing}')
        return t
                    
        
    
    def generate_nodes(self, node_pos):
        node = self._nodes[self._pos_to_node_id[node_pos]]
        # add the root node
        nodes = [self.make_node_point(
                        x=0,
                        y=0, 
                        label='root',
                        id=0,
                        parent_id=np.nan,
                        node=self._tree
                    )
                ]
        ypos = 0
        xpos = len(node.path) + 1

        top = None
        while node:
            children = sorted(node.children.values(), key=lambda x : x.name)
            # put selected node at the top
            if top:
                children.remove(top)
                # shift to the right slightly 
                nodes.append(self.make_node_point(
                        x=xpos + .2,
                        y=ypos, 
                        label=top.name,
                        id=top.id,
                        parent_id=node.id,
                        node=top
                ))
                ypos -= 3 # add larger gap between selected and the rest

            for c in children:
                nodes.append(self.make_node_point(
                        x=xpos,
                        y=ypos, 
                        label=c.name,
                        id=c.id,
                        parent_id=node.id,
                        node=c
                ))
                ypos -= 1
            

            top = node
            node = node.parent
            xpos -= 1
            ypos = 0
        
        return pd.DataFrame(nodes, columns=self._node_point_keys)

    def get_clicked_node(self, node_pos):
        return self._node_df.iloc[node_pos]['node']

    def add_nodes(self, fig, node_df):

        fig.update_xaxes(range=(-.25, node_df['x'].max() + .6))

        fig.add_trace(
            go.Scatter(
                x = node_df['x'],
                y = node_df['y'],
                text = node_df['label'],
                mode = 'markers+text',
                textposition='top right'
            )
        )

        return node_df


    def generate_links(self, node_id):
        pass

    def add_links(self, fig, node_df):

        node_df = node_df.set_index('id')

        for idx, row in node_df.iterrows():
            pid = row['parent_id']
            if np.isnan(pid):
                continue
            r1 = node_df.loc[pid]
            line = self.link_points((r1.x, r1.y), (row.x, row.y))
            fig.add_trace(line)

    
    def _add_node_pos(self, node_df):
        node_df['x'] = node_df['depth']
        node_df['x'].loc[node_df['selected']] += .2

        node_df = node_df.groupby('depth')\
                            .apply(lambda x : x.sort_values(['render_order', 'label'])\
                                                .assign(y=np.arange(0, -len(x), -1))
                                )\
                        .reset_index(drop=True)
        return node_df


    def create_figure(self, node_pos=0):

        fig = go.Figure(
                layout=self.generate_layout()
            )

        node = self.get_clicked_node(node_pos)
        self._render_tree.toggle_node(node.id)
        self._node_df = self._render_tree.render()
        
        self._node_df = self._add_node_pos(self._node_df)
            
        self.add_nodes(fig, self._node_df)

        self.add_links(fig, self._node_df)

        return fig






def read_tree():
    with open('tree-all.pickle', 'rb') as ifs:
        tree = pickle.load(ifs)

    return Tree(tree)


def create_app(tree):

    logger.info('creating app')

    app = dash.Dash()
    app.layout = html.Div([
        html.H1('Node link Diagram of tree'),

        dcc.Graph(
            id='tree',
            figure=tree.create_figure()
        ),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns')
    ])
    

    @app.callback(
            Output('tree', 'figure'),
            Input('tree', 'clickData'))
    def display_click_data(data):
        print(data)
        if data is None:
            return tree.create_figure()

        node_id = data['points'][0]['pointNumber']
        return tree.create_figure(node_id)
    return app




logger.info('reading tree')
tree = read_tree()

app = create_app(tree)

if __name__ == '__main__':
    logger.info('starting server')
    app.run_server(debug=True)  # Turn off reloader if inside Jupyter



