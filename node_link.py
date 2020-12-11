import plotly.graph_objects as go
import dash_table
import plotly.express as px
import pickle
import ast
from pprint import pformat
import numpy as np
import pandas as pd
from convert_tree import CompactNode
import traceback

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

import sys
from argparse import ArgumentParser

import logging
import logging.config


argp = ArgumentParser()
argp.add_argument('--debug', action='store_true')


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

CSV_DATA = pd.read_csv('all-nodes.csv')[['pathName', 'name', 'alsoCount', 'id']]\
                .set_index('id')



def dist(p1,p2):
    return np.sqrt((p1[0] - p2[0])**2 +  (p1[1] - p2[1])**2)

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
        if self._node.children:
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
            # node has children
            if c._node.children:
                render_order[-1] += 1
        




    
class Tree:

    def __init__(self, tree):
        self._tree = tree

        self._link_res = 30
        self._cos_y = np.cos(np.linspace(0, np.pi, self._link_res, endpoint=True, dtype=np.float32))


        # add root node
        self._render_tree = RenderNode(self._tree, None)
        self._node_df = self._render_tree.render()

        self._line_colors = [
                px.colors.qualitative.Safe[0],
                #px.colors.qualitative.Safe[1],
                #px.colors.qualitative.Safe[2],
                #px.colors.qualitative.Safe[4],
        ]

        self._x_selected_offset = .5
        self._y_selected_offset = 1

        self._also_line_dash = '2px'
        self._also_line_color = px.colors.qualitative.Pastel2[7]
        self._also_line_width = .5

        self._highlight_line_width = 5
        self._highlight_line_color = px.colors.qualitative.Vivid[8]

        self._highlight_also_line_width = 3
        self._highlight_also_line_color = px.colors.qualitative.Dark2[2]

        self._fig = None
        self._table = None
        self._bar_chart = go.Figure()
        self._mode = None

        self._highlighted_nodes = [
                None, None
        ]
    
    def generate_layout(self, node_df):
        max_nodes = node_df.groupby('depth').count().max().iat[0]
        max_depth = node_df['x'].max()

        return {
            'height' : max_nodes * 27,
            'width' : max_depth * 500 + 200,
            'showlegend' : False,
            'plot_bgcolor' : 'white',
            'xaxis' : {
                'showticklabels' : False
            },
            'yaxis' : {
                'showticklabels' : False
            },

        }

    def link_points(self, p1, p2, type='cos'):
        if type == 'cos':
            y = (self._cos_y  * (abs(p1[1] - p2[1]) / 2)) + ((p1[1] + p2[1]) / 2)
            if p2[1] > p1[1]:
                y = np.flip(y)

            x = np.linspace(p1[0], p2[0], len(y), dtype=np.float32)

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
                        mode='lines',
                        hoverinfo='none'
                    )

        elif type == 'arc':

            if p1[1] > p2[1]:
                p1, p2, = p2, p1
                
            arc = np.pi / 60
            # 120 degree arc
            p4 = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            r = dist(p4, p1) / np.cos(np.pi / 2  - (arc / 2))
            theta = np.arccos((p1[0] - p2[0]) / dist(p1, p2))
            theta_prime = theta - (np.pi / 2 - (arc / 2))
            p3 = (np.cos(theta_prime) * r + p1[0]), (np.sin(theta_prime) * r + p1[1])

            start = np.arccos((p2[0] - p3[0]) / r)
            #start = 2 * np.pi / 3
            thetas = np.linspace(start, start + arc, num=self._link_res, endpoint=True, dtype=np.float32)
            x = np.cos(thetas) * r + p3[0]
            y = np.sin(thetas) * r + p3[1]

            line = go.Scatter(
                        x = x,
                        y = y,
                        mode='lines',
                        hoverinfo='none'
                    )

            return line


        else:
            raise RuntimeError(f'unknown line type {type}')


    def link_parent_child(self, p, c):
            
            color = self._line_colors[c.render_order[-1] % len(self._line_colors)]
            lines = []
            if c['selected']:
                # extend the curve linearly 
                # to avoid collisions
                line = self.link_points((p.x, p.y), (c.x - self._x_selected_offset, c.prev_y))
                line.line.color = color
                lines.append(line)
                line = self.link_points((c.x - self._x_selected_offset, c.prev_y), (c.x, c.y))
                line.line.color = color
                lines.append(line)
            else:
                line = self.link_points((p.x, p.y), (c.x, c.y))
                line.line.color = color
                lines.append(line)

            return lines
    
    def link_also(self, node, node_df):
        lines = []
        for node_id, cnt in node.node.also.items():
            if node_id in node_df.index:
                other = node_df.loc[node_id]
                # don't like nodes with the same parent that are not selected
                if other.parent_id == node.parent_id and not other.selected:
                    continue
                if other.x == node.x:
                    line = self.link_points((other.x, other.y), (node.x, node.y), 'arc')
                else:
                    line = self.link_points((other.x, other.y), (node.x, node.y), 'linear')

                line.line.dash = self._also_line_dash
                if other.name in self._highlighted_nodes:
                    line.line.color = self._highlight_also_line_color
                    line.line.width = self._highlight_also_line_width
                else:
                    line.line.color = self._also_line_color
                    line.line.width = self._also_line_width


                lines.append(line)

        return lines


    def add_links(self, fig, node_df):

        for idx, row in node_df.iterrows():

            pid = row['parent_id']
            
            for l in self.link_also(row, node_df):
                fig.add_trace(l)

            if np.isnan(pid):
                continue

            parent = node_df.loc[pid]
            
            for l in self.link_parent_child(parent, row):
                fig.add_trace(l)

    
    def _make_hover_text(self, row):
        n = row['node']
        return f'''
{n.name}<br>
product count : {n.productCount}<br>
number of sub-categories : {len(n.children) if n.children else 0}
'''

    def _assign_node_y_pos(self, df, node_df):
        df = df.sort_values(['render_order', 'label'])
        avg_parent_y = node_df['y'].loc[df['parent_id'].unique()].mean()
        df['y'] = np.arange(0, -len(df), -1)
        df['y'] += avg_parent_y - df['y'].iat[len(df) // 2]
        # add gap between the subtrees 
        if len(df) > 1:
            df['y'] -= (np.diff(df.parent_id, prepend=df.parent_id.iat[0]) != 0).cumsum() * 2

        df = self._adjust_selected_ypos(df)
        return df
    
    def _adjust_selected_ypos(self, df):
        min_selected_y_dist = 2
        df['prev_y'] = df['y'].copy()
        y = df.loc[df.selected]['prev_y'].sort_values(ascending=False)

        if len(y) > 1:
            for i in range(1, len(y)):
                if np.abs(y.iat[i] - y.iat[i-1]) < min_selected_y_dist:
                    y.iat[i] = y.iat[i-1] - min_selected_y_dist

            # don't adjust the first node
            df.loc[y.index, 'y'] = y

        # adjust down a little more do avoid label collisions
        df.loc[df.selected, 'y'] -= self._y_selected_offset
        return df

        
    def _add_node_pos(self, node_df):
        node_df['prev_y'] = np.nan
        node_df['x'] = node_df['depth']
        # offset the selected nodes
        node_df.loc[node_df['selected'], 'x'] += self._x_selected_offset
        # assign root node 
        node_df.at[0, 'y'] = 0.0
        # for all other levels
        for d in node_df['depth'].unique()[1:]:
            s = node_df.loc[node_df.depth.eq(d)]
            node_df.loc[s.index] = self._assign_node_y_pos(s, node_df)


        node_df['sub_product_count'] = node_df['node'].apply(lambda x: x.subtreeProductCount)
        return node_df

    def add_nodes(self, fig, node_df):
        fig.update_xaxes(range=(self._x_selected_offset-.05, node_df['x'].max() + .6))

        
        scatter = go.Scatter(
                x = node_df['x'],
                y = node_df['y'],
                text = node_df['label'].apply(lambda x : f'<b>{x}</b>'),
                mode = 'markers+text',
                textposition = node_df['selected'].apply(lambda x : 'top center' if x else 'middle right')
            )

        #scatter.marker.symbol = 'circle-open'
        #scatter.marker.sizemode = 'area'
        scatter.marker.size = 18
        # color options
        scatter.marker.colorscale = 'Greens'
        scatter.marker.cmin = 0.0
        scatter.marker.cmax = 1.0
        
        colors = node_df['sub_product_count'].values.astype(np.float64)
        # limit colors 
        colors = np.clip(colors, 1, 100000)
        colors = np.log(colors)
        colors /= colors.max()
        scatter.marker.color = colors

        scatter.hovertext = node_df.apply(self._make_hover_text, axis=1)
        scatter.hoverinfo = 'text+x+y'
        #scatter.hoverlabel = node_df['product_count'].apply(str)


        fig.add_trace(scatter)

        return node_df

    def _highlight_paths(self, fig, node_ids, node_df):
        for nn in node_ids:
            # check for nulls or hidden nodes
            if nn is None or nn not in node_df.index:
                continue

            n = node_df.loc[nn]

            while n.parent_id in node_df.index:
                p = node_df.loc[n.parent_id]
                for l in self.link_parent_child(p, n):
                    l.line.width = self._highlight_line_width
                    l.line.color = self._highlight_line_color
                    fig.add_trace(l)
                n = p


            for l in self.link_also(n, node_df):
                l.line.width = self._highlight_also_line_width
                l.line.color = self._highlight_also_line_color
                fig.add_trace(l)
        

    def get_clicked_node(self, click_data):
        
        row = self._node_df.iloc[click_data['pointNumber']]
        if click_data['x'] == row.x and click_data['y'] == row.y:
            return row.node
        else:
            return None

    def create_table(self, highlight_nodes, node_df):

        NAME = 'Name'
        PATH = 'Path'
        SCAT_PROD_CNT = 'Subcategory Product Count'
        N_SCAT = 'Number of Subcategories'
        PROD_CNT = 'Product Count'
        D = 'Depth'
        N_CX_LISTED_CAT = 'Number of Categories with Shared Products'
        N_SHARED_PRODS = 'Number of Shared Products'
        P_SHARED_PRODS = 'Percent of Products Shared'
        # cannot get this info
       # N_CX_LISTED = 'Number of Cross Listed Products'
       # PC_CX_LISTED = 'Percent of Products Cross Listed'
        df = pd.DataFrame(
                columns= ['', 'Node 1' , 'Node 2'],
                index = [NAME,
                        PATH,
                        SCAT_PROD_CNT,
                        N_SCAT,
                        PROD_CNT,
                        D,
                        N_CX_LISTED_CAT,
                        N_SHARED_PRODS,
                        P_SHARED_PRODS
                    ]
            )


        for i, n_id in enumerate(highlight_nodes):
            if n_id is None or n_id not in node_df.index:
                continue

            n = node_df.loc[n_id]
            
            col = df.columns[i+1]
            df.loc[NAME, col] = n.node.name
            df.loc[PATH, col] = u' \u2794 '.join(n.node.path)
            df.loc[SCAT_PROD_CNT, col] = n.node.subtreeProductCount
            df.loc[N_SCAT, col] = len(n.node.children)
            df.loc[PROD_CNT, col] = n.node.productCount
            df.loc[D, col] = n.depth
            df.loc[N_CX_LISTED_CAT, col] = CSV_DATA.at[n.node.id, 'alsoCount'] 
        
        if all([i in node_df.index for i in highlight_nodes]):
            n0 = node_df.loc[highlight_nodes[0]]
            olap = n0.node.also[highlight_nodes[1]]
            df.loc[N_SHARED_PRODS] = olap
            df.loc[P_SHARED_PRODS] = df.loc[N_SHARED_PRODS] / df.loc[PROD_CNT]


        df[''] = df.index


    
        cols = [{'id' : c, 'name' : c} for c in df.columns]
        data = df.to_dict('records')
        return cols, data
    


    def create_bar_df(self, n_id, node_df):
            n = node_df.loc[n_id]
            bar_df = pd.DataFrame({
                'count' : list(n.node.also.values())
                },
                index = list(n.node.also.keys())
            )
            bar_df['path'] = CSV_DATA['pathName'].loc[bar_df.index].apply(ast.literal_eval)
            bar_df['name'] = CSV_DATA['name'].loc[bar_df.index]
            bar_df['label'] = bar_df['name'] + ' : ' + np.array(list(map(str, bar_df.index)))
            bar_df['id'] = bar_df.index
            bar_df = bar_df.set_index('label')
            

            bar_df['percent'] = bar_df['count'] / n.node.productCount if n.node.productCount > 0.0  else 0.0

            return bar_df

    def create_bar_fig(self, bar_df, name, color):
        bar_df = bar_df.sort_values('percent')
        bar = go.Bar(
                name = name,
                y = bar_df.index,
                x = bar_df['percent'],
                text = bar_df['percent'].apply(lambda x : f'{x:.3f}'),
                textposition='outside',
                hovertext=bar_df.apply(self.create_bar_chart_hover, axis=1),
                hoverinfo='x+text',
                orientation = 'h',
                marker_color = color
                #width=20

        )

        layout = go.Layout(
                title = {'text' : name},
                xaxis = {"mirror" : "allticks", 'side': 'top', 'title' : {'text' : f'Percent of {name} Products Shared'}},
                height = max(10, len(bar_df) * 25),
        )   
        return go.Figure(
                data = [bar],
                layout=layout
            )
    
    def create_bar_chart_hover(self, row):
        p = ' \u2794 '.join(row.path)
        return f'''
Path : {p}<br>
Number of Products Shared : {row.count}
'''


    def create_grouped_bar_fig(self, bar_dfs, colors):
        suffixes = ['', '2']
        joined = bar_dfs[0].join(bar_dfs[1],
                                    how='inner',
                                    lsuffix=suffixes[0],
                                    rsuffix=suffixes[1]
                            )

        if len(joined) == 0:
            return go.Figure()

        joined['sort_key'] = joined[['percent' + s for s in suffixes]].max(axis=1)
        joined = joined.sort_values('sort_key')
        data = []
        for i, suffix in enumerate(suffixes):
            bar = go.Bar(
                    name = f'Node {i+1}',
                    y = joined.index,
                    x = joined[f'percent{suffix}'],
                    text = joined[f'percent{suffix}'].apply(lambda x : f'{x:.3f}'),
                    textposition='outside',
                    hovertext=joined.apply(self.create_bar_chart_hover, axis=1),
                    hoverinfo='x+text',
                    orientation = 'h',
                    marker_color = colors[i]
                    #width=20

            )

            data.append(bar)


        layout = go.Layout(
                title = {'text' : 'Overlap'},
                xaxis = {"mirror" : "allticks", 'side': 'top', 'title' : {'text' : 'Percent of Products Shared'}},
                height = max(10, len(joined) * 50),
                barmode='group'
        )   
        return go.Figure(
                data = data,
                layout=layout
            )


    def create_bar_charts(self, highlight_nodes, node_df):
        
        dfs = []
        for i, n_id in enumerate(highlight_nodes):
            if n_id is None or n_id not in node_df.index:
                dfs.append(None)
            else:
                dfs.append(self.create_bar_df(n_id, node_df))

        bar_colors = [
                px.colors.qualitative.Safe[0],
                px.colors.qualitative.Safe[1],
        ]

        figs = [
                self.create_bar_fig(dfs[0], 'Node 1', bar_colors[0]) if dfs[0] is not None else go.Figure(),
                self.create_grouped_bar_fig(dfs, bar_colors) if all(d is not None for d in dfs) else go.Figure(),
                self.create_bar_fig(dfs[1], 'Node 2', bar_colors[1]) if dfs[1] is not None else go.Figure(),
        ]
        xmax = 0.0
        max_bars = 0
        for df in dfs:
            if df is not None:
                xmax = max(xmax, df['percent'].max())
                max_bars = max(max_bars, len(df))
        

        for f in figs:
            f.update_xaxes(range=(0, xmax * 1.15))
            #f.update_layout(height=max(500, max_bars*25))

        return figs


    def create_figure(self, click_data, click_mode):
        if click_data is None or self._click_mode != click_mode:
            self._click_mode = click_mode
            if self._fig:
                return (self._fig, *self._table, *self._bar_charts)
            node = self._tree
        else:
            node = self.get_clicked_node(click_data)

        if node is None:
            return (self._fig, *self._table, *self._bar_charts)
        

        if click_mode >= 0:
            # highlight node
            self._highlighted_nodes[click_mode] = node.id

        else:
            # select or deselect node, expand or collapse
            self._render_tree.toggle_node(node.id)

        # DONT TOUCH THESE LINES
        fig = go.Figure()


        self._node_df = self._render_tree.render()
        self._node_df = self._add_node_pos(self._node_df)
            
        self.add_links(fig, self._node_df)
        self._highlight_paths(fig, self._highlighted_nodes, self._node_df)
        self.add_nodes(fig, self._node_df)

        fig.update_layout(self.generate_layout(self._node_df))
        self._fig = fig
        # create basic table
        self._table = self.create_table(self._highlighted_nodes, self._node_df)
        # create bar chart
        self._bar_charts = self.create_bar_charts(self._highlighted_nodes, self._node_df)

        return (self._fig, *self._table, *self._bar_charts)



def read_tree():
    with open('compact_tree.pickle', 'rb') as ifs:
        tree = pickle.load(ifs)

    return Tree(tree)


def create_app(tree):

    logger.info('creating app')

    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
        html.H1('Node link Diagram of tree'),
        # the table to display data about the nodes
        # select the different click actions
        html.Div([
                html.H4('Detail View'),
                dash_table.DataTable(
                    id='table',
                    style_cell = {
                        'whiteSpace' : 'normal',
                        'height'  : 'auto',
                        'textAlign' : 'left',
                    },
                    style_table = {
                        'maxWidth' : '1500px',
                    },
                    columns = [],
                    data = [],
                    style_data_conditional = [
                        {'if': {'column_id': ''}, 'textAlign': 'right'}
                    ],
                )
            ],
            #style={'text-align' : 'center'}
        ),
        html.Div([
            html.Div(
                dcc.Graph(
                    id='bar-chart-0',
                    figure=go.Figure(),
                ),
                className="four columns"
            ),
            html.Div([
                dcc.Graph(
                    id='bar-chart-1',
                    figure=go.Figure(),
                ),],
                className="four columns"
            ),
            html.Div([
                dcc.Graph(
                    id='bar-chart-2',
                    figure=go.Figure(),
                ),],
                className="four columns"
            ),
            ],
            style={
                'max-height' : '500px',
                'overflow-y' : 'scroll',
                'position' : 'relative',
            },
            className='row'
        ),


        html.Div([
                html.H4('Click Action'),
                dcc.RadioItems(
                    options=[
                        {'label' : 'Expand/Collapse', 'value' : -1},
                        {'label' : 'Select Node 1', 'value' : 0},
                        {'label' : 'Select Node 2', 'value' : 1}
                    ],
                    value = -1, 
                    labelStyle={'display': 'inline-block'},
                    id='click-mode'
                )
            ],
            style={'text-align' : 'center'}
        ),
        html.Div(
            dcc.Graph(
                id='tree',
                figure=go.Figure()
            ),
            style={
                'max-height' : '1500px',
                'overflow-y' : 'scroll',
                'position' : 'relative',
            },
            className='row'
        )

    ])


    @app.callback(
            [Output('tree', 'figure'),
             Output('table', 'columns'),
             Output('table', 'data'),
             Output('bar-chart-0', 'figure'),
             Output('bar-chart-1', 'figure'),
             Output('bar-chart-2', 'figure'),
             ],
            [Input('tree', 'clickData'),
             Input('click-mode', 'value')])
    def display_click_data(click_data, click_mode):
        print(f'click_data : {pformat(click_data)}')
        if click_data is None:
            return tree.create_figure(None, click_mode)
        else:
            data = click_data['points'][0]
            return tree.create_figure(data, click_mode)




    return app




logger.info('reading tree')
try:
    tree = read_tree()
except:
    traceback.print_exc()

app = create_app(tree)
server = app.server
if __name__ == '__main__':
    args = argp.parse_args(sys.argv[1:])
    logger.info('starting server')
    app.run_server(debug=args.debug)  # Turn off reloader if inside Jupyter



