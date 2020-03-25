#!/usr/bin/env python
#
# Copyright (c) 2020 Björn Gottschall <bjorn.gottschall@ntnu.no>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import shutil
import subprocess
import tempfile
import argparse
import plotly
import plotly.graph_objects as go

parser = argparse.ArgumentParser(description="plots the contained figure")
parser.add_argument("--font-size", help="font size (default %(default)s)", type=int, default=12)
parser.add_argument("--orca", help="path to plotly orca (https://github.com/plotly/orca)", type=str, default=None)
parser.add_argument("--width", help="width of output file (default %(default)s)", type=int, default=1000)
parser.add_argument("--height", help="height of output (default %(default)s)", type=int, default=None)
parser.add_argument("--output", help="output file (html, png, jpg, pdf...) (default %(default)s)", type=str, nargs="+", default=[])
parser.add_argument("--no-output", help="no output, just open an html plot", action="store_true", default=False)
parser.add_argument("--quiet", help="do not automatically open output file", action="store_true", default=False)

args = parser.parse_args()


def checkOrca(orca = 'orca'):
    if orca is not None:
        orca = shutil.which(orca)
    if orca is None:
        raise Exception('Could not find plotly orca please provide it via --orca (https://github.com/plotly/orca)')
    orcaOutput = subprocess.run([orca, '--help'], stdout=subprocess.PIPE, check=True)
    if 'Plotly\'s image-exporting utilities' not in orcaOutput.stdout.decode():
       raise Exception(f'Invalid orca version {orca}. Please provide the correct version via --orca (https://github.com/plotly/orca)')
    return orca


def exportFigure(fig, width, height, exportFile, orca = 'orca'):
    if exportFile.endswith('.html'):
        plotly.offline.plot(fig, filename=exportFile, auto_open=False)
        return
    else:
        tmpFd, tmpFile = tempfile.mkstemp()
        try:
            exportFile = os.path.abspath(exportFile)
            exportDir = os.path.dirname(exportFile)
            exportFilename = os.path.basename(exportFile)
            _, fileExtension = os.path.splitext(exportFilename)
            fileExtension = fileExtension.lstrip('.')

            go.Figure(fig).write_json(tmpFile)
            cmd = [orca, 'graph', tmpFile, '--output-dir', exportDir, '--output', exportFilename, '--format', fileExtension, '--mathjax', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js']
            if width is not None:
                cmd.extend(['--width', f'{width}'])
            if height is not None:
                cmd.extend(['--height', f'{height}'])
            subprocess.run(cmd, check=True)
        finally:
            os.remove(tmpFile)



plotly.io.templates.default = 'plotly_white'
fig = plotly.subplots.make_subplots(
    cols=1,
    rows=1,
    shared_xaxes=False,
    shared_yaxes=False,
    y_title=None,
    x_title=None,
    vertical_spacing=0.0,
    horizontal_spacing=0.0,
    specs=[
    [{'rowspan': 1, 'colspan': 1, 'secondary_y': False},],
    ],
)
fig.add_trace(go.Bar(
    name='length',
    legendgroup='length',
    orientation='h',
    marker_color=['#00ff00', '#00ffff', '#084a91', '#00ff00', '#00ff00', '#084a91', '#ff0000', '#00ff00', '#00ffff', '#00ff00'],
    marker_line_color='#222',
    marker_line_width=1,
    y=['Morning Sleep', 'Breakfast', 'Work', 'Break', 'Lunch', 'Work', 'Exercise', 'Post Workout Rest', 'Dinner', 'Evening Sleep'],
    x=['1970-01-01 06:00:00', '1970-01-01 01:30:00', '1970-01-01 02:25:00', '1970-01-01 00:30:00', '1970-01-01 01:00:00', '1970-01-01 04:00:00', '1970-01-01 01:00:00', '1970-01-01 00:30:00', '1970-01-01 01:00:00', '1970-01-01 02:59:00'],
    width=1.5,
    text=['Sleep', 'Food', 'Brain', 'Rest', 'Food', 'Brain', 'Cardio', 'Rest', 'Food', 'Sleep'],
    textposition='inside',
    base=['2020-01-01 00:00:00', '2020-01-01 07:00:00', '2020-01-01 09:00:00', '2020-01-01 11:30:00', '2020-01-01 12:00:00', '2020-01-01 13:00:00', '2020-01-01 17:30:00', '2020-01-01 18:30:00', '2020-01-01 19:00:00', '2020-01-01 21:00:00'],
    opacity=None,
), col=1, row=1, secondary_y=False)


# Subplot specific options:
fig.update_yaxes(type='-', rangemode='normal', col=1, row=1, secondary_y=False)
fig.update_xaxes(type='date', rangemode='normal', col=1, row=1)
fig.update_yaxes(showline=False, linewidth=0, linecolor='rgba(0,0,0,0)', col=1, row=1, secondary_y=False)
# fig.update_yaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row=1, col=1, secondary_y=False)
# fig.update_xaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row=1, col=1)
# fig.update_yaxes(title_text='None', col=1, row=1, secondary_y=False)
# fig.update_xaxes(title_text='None', col=1, row=1)
fig.update_yaxes(tickformat='', ticksuffix='', tickprefix='', col=1, row=1, secondary_y=False)
fig.update_xaxes(tickformat='', ticksuffix='', tickprefix='', col=1, row=1)
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col=1, row=1, secondary_y=False)
# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col=1, row=1)
# fig.update_yaxes(range=[None, None], col=1, row=1, secondary_y=False)
# fig.update_xaxes(range=[None, None], col=1, row=1, secondary_y=False)




# Global modes and paramters:
fig.update_layout(barmode='stack', boxmode='overlay', violinmode='overlay', violingap=0.3, violingroupgap=0.3)

# Layout Legend
fig.update_layout(showlegend=None)
# fig.update_layout(legend_yanchor='auto')
fig.update_layout(legend_xanchor='left')
fig.update_layout(legend=dict(x=1.02, y=1.0, orientation='v', bgcolor='rgba(255,255,255,0)'))

# Layout Plot and Background
fig.update_layout(paper_bgcolor='rgba(255, 255, 255, 0)', plot_bgcolor='rgba(255, 255, 255, 0)')
fig.update_layout(margin=dict(t=0, l=None, r=0, b=None, pad=0))

# Plot Font
fig.update_layout(font=dict(family='"Open Sans", verdana, arial, sans-serif', size=args.font_size, color='#000'))

# Execute addon file if found
filename, fileext = os.path.splitext(__file__)
if (os.path.exists(f'{filename}_addon{fileext}')):
    exec(open(f'{filename}_addon{fileext}').read())

if args.orca is None and os.getenv('PLOTLY_ORCA') is not None:
    args.orca = os.getenv('PLOTLY_ORCA')

# An initial orca version is provided by the plot author
if args.orca is None:
    args.orca = '/opt/plotly-orca/orca'

if not args.output or args.no_output:
    fig.show()
else:
    if not args.quiet:
        openWith = None
        for app in ['xdg-open', 'open', 'start']:
            if shutil.which(app) is not None:
                openWith = app
                break
        if openWith is None:
            args.quiet = True

    orca = checkOrca(args.orca);
    for output in args.output:
        exportFigure(fig, args.width, args.height, output, args.orca)
        print(f'Saved to {output}')
        if not args.quiet:
            try:
                subprocess.check_call([openWith, output], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                print(f'Could not open {output}!')
