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





class Tree:

    def __init__(self, tree):
        self._tree = tree
        self._nodes = self._get_nodes()
        self.labels = self._get_labels(self._nodes)
        self._pos_to_node_id = [0]
    
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
            'height' : 2000
        }

    def link_points(self, p1, p2):
        return go.Scatter(
                    x = [p1[0], p2[0]],
                    y = [p1[1], p2[1]],
                    mode='lines'
                )



                    
        
    
    def generate_nodes(self, node_pos):
        node = self._nodes[self._pos_to_node_id[node_pos]]
        # add the root node

        nodes = [(0, 0, 'root', 0, np.nan, self._tree)]
        ypos = 0
        xpos = len(node.path) + 1

        top = None
        while node:
            children = sorted(node.children.values(), key=lambda x : x.name)
            # put selected node at the top
            if top:
                children.remove(top)
                # shift to the right slightly 
                nodes.append((xpos + .2, ypos, top.name, top.id, node.id, top))
                ypos -= 3 # add larger gap between selected and the rest

            for c in children:
                nodes.append((xpos, ypos, c.name, c.id, node.id, c))
                ypos -= 1
            

            top = node
            node = node.parent
            xpos -= 1
            ypos = 0
        
        return pd.DataFrame(nodes, columns=['x', 'y', 'label', 'id', 'parent_id', 'node'])



    def add_nodes(self, fig, node_pos):
        node = self._nodes[self._pos_to_node_id[node_pos]]
        
        xpos = len(node.path) + 1

        fig.update_xaxes(range=(-.25, xpos + .6))

        node_df = self.generate_nodes(node_pos)

        fig.add_trace(
            go.Scatter(
                x = node_df['x'],
                y = node_df['y'],
                text = node_df['label'],
                mode = 'markers+text',
                textposition='middle right'
            )
        )
        # update mapping for future lookup
        self._pos_to_node_id = node_df['id'].values

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




    def create_figure(self, node_pos=0):

        fig = go.Figure(
                layout=self.generate_layout()
            )
        
        node_df = self.add_nodes(fig, node_pos)

        self.add_links(fig, node_df)

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



