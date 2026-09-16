from graphviz import Digraph

def draw_network_struct():
    dot = Digraph('Network_Structure', comment='Dual Branch Network')
    dot.attr(rankdir='LR', size='12,8', dpi='300')
    
    # 节点通用设置
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    
    # 输入层
    dot.node('Input_RGB', 'RGB 单帧\n3×224×224', fillcolor='#E1F5FE')
    dot.node('Input_Dyn', '动态成像\n3×224×224', fillcolor='#E1F5FE')
    
    # 主干网络 (RGB分支)
    with dot.subgraph(name='cluster_RGB') as c:
        c.attr(label='RGB 分支 (Spatial)', style='dashed')
        c.node('ResNet_RGB', 'ResNet-18 (Layer4)\n(JAFFE 预训练)', fillcolor='#FFF9C4')
        c.node('SPP_RGB', 'SPP (Levels:1,2,4)\nOut: 10752 dim', fillcolor='#FFE0B2')
        c.node('Proj_RGB', '线性投影\n10752 → 512', fillcolor='#FFCCBC')
        c.edge('Input_RGB', 'ResNet_RGB')
        c.edge('ResNet_RGB', 'SPP_RGB')
        c.edge('SPP_RGB', 'Proj_RGB')

    # 主干网络 (Dynamic分支)
    with dot.subgraph(name='cluster_Dyn') as c:
        c.attr(label='动态分支 (Temporal)', style='dashed')
        c.node('ResNet_Dyn', 'ResNet-18 (Layer4)\n(JAFFE 预训练)', fillcolor='#FFF9C4')
        c.node('SPP_Dyn', 'SPP (Levels:1,2,4)\nOut: 10752 dim', fillcolor='#FFE0B2')
        c.node('Proj_Dyn', '线性投影\n10752 → 512', fillcolor='#FFCCBC')
        c.edge('Input_Dyn', 'ResNet_Dyn')
        c.edge('ResNet_Dyn', 'SPP_Dyn')
        c.edge('SPP_Dyn', 'Proj_Dyn')

    # 融合与分类
    dot.node('Concat', '特征拼接 (Concat)\n512 + 512 = 1024', fillcolor='#F8BBD0', shape='component')
    dot.node('Dropout', 'Dropout (p=0.5)', fillcolor='#E1BEE7')
    dot.node('FC', '全连接层 (FC)\n1024 → 3', fillcolor='#D1C4E9')
    dot.node('Output', '输出分类\n(Pos, Neg, Sur)', fillcolor='#C5CAE9', shape='ellipse')

    # 连接分支到融合层
    dot.edge('Proj_RGB', 'Concat')
    dot.edge('Proj_Dyn', 'Concat')
    dot.edge('Concat', 'Dropout')
    dot.edge('Dropout', 'FC')
    dot.edge('FC', 'Output')

    dot.render('Figure1_Network_Structure', format='png', cleanup=True)
    print("图1已生成: Figure1_Network_Structure.png")

if __name__ == '__main__':
    draw_network_struct()