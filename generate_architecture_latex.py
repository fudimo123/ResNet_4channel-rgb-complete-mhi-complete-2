
def generate_latex():
    latex_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{./layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} % for 3d coordinates

\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}

\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

% --- Stream 1: RGB (Top Branch) ---
% Input RGB
\node[canvas is zy plane at x=0] (rgb_input) at (0, 3, 0) {\includegraphics[width=2cm,height=2cm]{rgb_dummy.jpg}};
\node[yshift=0.5cm] at (rgb_input.north) {\textbf{RGB Input}};

% ResNet Backbone (Simplified as a block)
\pic[shift={(2, 3, 0)}] at (0,0,0) {Box={
    name=rgb_resnet,
    caption=ResNet-18 (RGB),
    xlabel={512},
    zlabel={7},
    fill=\ConvColor,
    height=10,
    width=2,
    depth=10
    }
};

% SPP Block
\pic[shift={(1.5,0,0)}] at (rgb_resnet-east) {Box={
    name=rgb_spp,
    caption=SPP,
    fill=\PoolColor,
    height=6,
    width=2,
    depth=6
    }
};

% Flatten/FC
\pic[shift={(1.5,0,0)}] at (rgb_spp-east) {Box={
    name=rgb_fc,
    caption=Features,
    xlabel={512},
    fill=\FcColor,
    height=1,
    width=4,
    depth=1
    }
};

% Connections
\draw [connection]  (rgb_resnet-east)    -- node {\midarrow} (rgb_spp-west);
\draw [connection]  (rgb_spp-east)       -- node {\midarrow} (rgb_fc-west);


% --- Stream 2: Dynamic (Bottom Branch) ---
% Input Dynamic
\node[canvas is zy plane at x=0] (dyn_input) at (0, -3, 0) {\includegraphics[width=2cm,height=2cm]{dyn_dummy.jpg}};
\node[yshift=-2.5cm] at (dyn_input.south) {\textbf{Dynamic Input}};

% ResNet Backbone
\pic[shift={(2, -3, 0)}] at (0,0,0) {Box={
    name=dyn_resnet,
    caption=ResNet-18 (Dynamic),
    xlabel={512},
    zlabel={7},
    fill=\ConvColor,
    height=10,
    width=2,
    depth=10
    }
};

% SPP Block
\pic[shift={(1.5,0,0)}] at (dyn_resnet-east) {Box={
    name=dyn_spp,
    caption=SPP,
    fill=\PoolColor,
    height=6,
    width=2,
    depth=6
    }
};

% Flatten/FC
\pic[shift={(1.5,0,0)}] at (dyn_spp-east) {Box={
    name=dyn_fc,
    caption=Features,
    xlabel={512},
    fill=\FcColor,
    height=1,
    width=4,
    depth=1
    }
};

% Connections
\draw [connection]  (dyn_resnet-east)    -- node {\midarrow} (dyn_spp-west);
\draw [connection]  (dyn_spp-east)       -- node {\midarrow} (dyn_fc-west);


% --- Fusion ---
% Concatenate
\pic[shift={(2, 0, 0)}] at (rgb_fc-east) {Box={
    name=concat,
    caption=Concat (1024),
    fill=\SumColor,
    height=2,
    width=1,
    depth=15
    }
};

% Dropout
\pic[shift={(1.5, 0, 0)}] at (concat-east) {Box={
    name=dropout,
    caption=Dropout,
    fill=\UnpoolColor,
    height=2,
    width=1,
    depth=15,
    opacity=0.5
    }
};

% FC Output
\pic[shift={(1.5, 0, 0)}] at (dropout-east) {Box={
    name=output_fc,
    caption=FC (3),
    xlabel={3},
    fill=\SoftmaxColor,
    height=1,
    width=1,
    depth=3
    }
};

% Fusion Connections
% Connect both branches to Concat
\draw [connection]  (rgb_fc-east)        -- node {\midarrow} (concat-west);
\draw [connection]  (dyn_fc-east)        -- node {\midarrow} (concat-west);

\draw [connection]  (concat-east)        -- node {\midarrow} (dropout-west);
\draw [connection]  (dropout-east)       -- node {\midarrow} (output_fc-west);

\end{tikzpicture}
\end{document}
"""
    
    with open('architecture_plot.tex', 'w') as f:
        f.write(latex_code)
    print("LaTeX code generated in 'architecture_plot.tex'.")
    print("Please use PlotNeuralNet library (https://github.com/HarisIqbal88/PlotNeuralNet) to compile it.")
    print("Note: You need the 'layers' folder from PlotNeuralNet in the same directory to compile this.")

if __name__ == "__main__":
    generate_latex()
