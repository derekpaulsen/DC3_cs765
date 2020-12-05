import plotly.graph_objects as go
import plotly.express as px
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
            if c.toggle_node(id):
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

        return df.set_index('id')


    def _render(self, points, depth, render_order):
        points[self.id] = self.make_node_point(
                    label=self.name,
                    id=self.id,
                    depth=depth,
                    parent_id=self._parent.id if self._parent else np.nan,
                    parent_label= self._parent.name if self._parent else '',
                    node=self._node,
                    selected=True,
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

            if len(c._node.children) > 0:
                render_order[-1] += 1
        




    
class Tree:

    def __init__(self, tree):
        self._tree = tree
        self._pos_to_node_id = [0]

        self._link_res = 30
        self._cos_y = np.cos(np.linspace(0, np.pi, self._link_res, endpoint=True))
        self._x_selected_offset = .5

        # add root node
        self._render_tree = RenderNode(self._tree, None)
        self._node_df = self._render_tree.render()

        self._line_colors = [
                px.colors.qualitative.Pastel[0],
                #px.colors.qualitative.Pastel[2],
                px.colors.qualitative.Pastel[4],
                px.colors.qualitative.Pastel[9],
        ]

        self._also_line_dash = '2px'
        self._also_line_color = 'grey'
        self._also_line_width = .5
    
    def generate_layout(self, node_df):
        max_nodes = node_df.groupby('depth').count().max().iat[0]

        return {
            'height' : max_nodes * 27,
            'showlegend' : False,
            'plot_bgcolor' : 'white'

        }

    def link_points(self, p1, p2, type='cos'):
        if type == 'cos':
            y = (self._cos_y  * (abs(p1[1] - p2[1]) / 2)) + ((p1[1] + p2[1]) / 2)
            if p2[1] > p1[1]:
                y = np.flip(y)

            x = np.linspace(p1[0], p2[0], len(y))

            line = go.Scatter(
                        x = x,
                        y = y,
                        mode='lines',
                        hoverinfo='none'
                    )
            return line
        elif type == 'linear':
            return go.Scatter(
                        x = [p1[0], p2[0]],
                        y = [p1[1], p2[1]],
                        mode='lines'
                    )
        else:
            raise RuntimeError(f'unknown line type {type}')

    
    def add_links(self, fig, node_df):

        for idx, row in node_df.iterrows():
            pid = row['parent_id']
            
            for node_id, cnt in row.node.also.items():
                if node_id in node_df.index:
                    other = node_df.loc[node_id]
                    line = self.link_points((other.x, other.y), (row.x, row.y), 'linear')
                    line.line.color = self._also_line_color
                    line.line.dash = self._also_line_dash
                    line.line.width = self._also_line_width

                    fig.add_trace(line)

            if np.isnan(pid):
                continue

            r1 = node_df.loc[pid]
            
            color = self._line_colors[row.render_order[-1] % len(self._line_colors)]
            if row['selected']:
                # extend the curve linearly 
                # to avoid collisions
                line = self.link_points((r1.x, r1.y), (row.x - self._x_selected_offset, row.y))
                line.line.color = color
                fig.add_trace(line)
                line = self.link_points((row.x - self._x_selected_offset, row.y), (row.x, row.y))
                line.line.color = color
                fig.add_trace(line)
            else:
                line = self.link_points((r1.x, r1.y), (row.x, row.y))
                line.line.color = color
                fig.add_trace(line)
    

    
    def _make_hover_text(self, row):
        n = row['node']
        return f'''
{n.name}<br>
subtree product count : {n.subtreeProductCount}<br>
number of children : {len(n.children)}
'''

    def _assign_node_y_pos(self, df):
        df = df.sort_values(['render_order', 'label'])
        df['y'] = np.arange(0, -len(df), -1)
        if len(df) > 1:
            df['y'] -= np.diff(df.render_order.apply(lambda x : x[-1]), prepend=0).cumsum() - 1
        return df

    def _add_node_pos(self, node_df):
        node_df['x'] = node_df['depth']
        node_df.loc[node_df['selected'], 'x'] += self._x_selected_offset
        node_df = node_df.reset_index()\
                         .groupby('depth')\
                            .apply(self._assign_node_y_pos)\
                        .reset_index(drop=True)\
                        .set_index('id')

        node_df['product_count'] = node_df['node'].apply(lambda x: x.productCount)
        node_df.at[0, 'y'] = node_df['y'].loc[node_df.depth == 1].median()
        return node_df

    def add_nodes(self, fig, node_df):

        fig.update_xaxes(range=(.25, node_df['x'].max() + .6))

        
        scatter = go.Scatter(
                x = node_df['x'],
                y = node_df['y'],
                text = node_df['label'],
                mode = 'markers+text',
                textposition=node_df['selected'].apply(lambda x : 'top center' if x else 'middle right')
            )

        #scatter.marker.symbol = 'circle-open'
        #scatter.marker.sizemode = 'area'
        scatter.marker.size = 18
        scatter.marker.colorscale = 'Reds'
        scatter.marker.colorscale = 'Reds'
        scatter.marker.cmin = 0.0
        scatter.marker.cmax = 1.0

        colors = node_df['product_count'].values
        colors[colors == 0] = 1
        colors = np.log(colors * 2)
        colors /= colors.max()
        scatter.marker.color = colors

        scatter.hovertext = node_df.apply(self._make_hover_text, axis=1)
        scatter.hoverinfo = 'text'
        #scatter.hoverlabel = node_df['product_count'].apply(str)


        fig.add_trace(scatter)

        return node_df

    def get_clicked_node(self, click_data):
        
        row = self._node_df.iloc[click_data['pointNumber']]
        if click_data['x'] == row.x and click_data['y'] == row.y:
            return row.node
        else:
            return None

    def create_figure(self, click_data=None):
        if click_data is None:
            node = self._tree
        else:
            node = self.get_clicked_node(click_data)

        if node is None:
            return self._fig

        fig = go.Figure()
        self._render_tree.toggle_node(node.id)
        self._node_df = self._render_tree.render()
        
        self._node_df = self._add_node_pos(self._node_df)
            
        self.add_links(fig, self._node_df)
        self.add_nodes(fig, self._node_df)

        fig.update_layout(self.generate_layout(self._node_df))
        self._fig = fig

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
            return tree.create_figure(None)
        
        data = data['points'][0]
        return tree.create_figure(data)
    return app




logger.info('reading tree')
tree = read_tree()

app = create_app(tree)

if __name__ == '__main__':
    logger.info('starting server')
    app.run_server(debug=True)  # Turn off reloader if inside Jupyter



