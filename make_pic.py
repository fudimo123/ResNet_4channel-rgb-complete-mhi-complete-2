from graphviz import Digraph
import os

def create_micro_expression_architecture():
    # 初始化图表，设置自上而下的布局，正交连线（直线转折），高DPI
    dot = Digraph(comment='ME_Recognition_Architecture', format='pdf')
    dot.attr(rankdir='TB', splines='ortho', dpi='300', charset='UTF-8')
    
    # 全局字体和节点设置 (SimHei用于显示中文，需确保系统有该字体，否则可改为其他中文字体)
    # 如果在Linux/Mac下，可能需要改为 'PingFang SC' 或 'Heiti'
    font_name = 'SimHei' 
    dot.attr('node', shape='box', style='filled', fontname=font_name, fontsize='10', margin='0.2')
    dot.attr('edge', fontname=font_name, fontsize='9')
    dot.attr('graph', fontname=font_name, labelloc='t', fontsize='16')

    # 设置图标题
    dot.attr(label=r'图标题：基于动态成像与SPP的特征级晚期融合微表情识别网络结构图\n\n')

    # 定义颜色方案 (学术蓝、学术红/橙、中性灰)
    color_rgb_bg = '#E6F3FF' # 浅蓝
    color_rgb_border = '#336699'
    color_dyn_bg = '#FFF0E6' # 浅橙
    color_dyn_border = '#CC6600'
    color_shared = '#F0F0F0'
    color_head = '#E8F6F3'

    # --- 左路分支：RGB Input ---
    with dot.subgraph(name='cluster_rgb') as c:
        c.attr(label='左路分支 (RGB Spatial)', style='dashed', color=color_rgb_border, fontcolor=color_rgb_border)
        
        # 输入与预处理
        c.node('rgb_input', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>左路输入</B></TD></TR>
            <TR><TD>RGB 单帧图像</TD></TR>
            <TR><TD><I>MTCNN 检测与裁剪</I></TD></TR>
            <TR><TD><FONT COLOR="blue">3×224×224</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_rgb_bg, color=color_rgb_border)

        # ResNet Backbone
        c.node('rgb_backbone', '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" COLOR="#336699">
            <TR><TD COLSPAN="2"><B>ResNet-18 (截断至 layer4)</B></TD></TR>
            <TR><TD COLSPAN="2"><I>加载 JAFFE 预训练权重</I></TD></TR>
            <TR><TD>conv1 → bn1 → relu</TD><TD ALIGN="RIGHT">64×112×112</TD></TR>
            <TR><TD>maxpool</TD><TD ALIGN="RIGHT">64×56×56</TD></TR>
            <TR><TD>layer1 (BasicBlock)</TD><TD ALIGN="RIGHT">64×56×56</TD></TR>
            <TR><TD>layer2 (BasicBlock)</TD><TD ALIGN="RIGHT">128×28×28</TD></TR>
            <TR><TD>layer3 (BasicBlock)</TD><TD ALIGN="RIGHT">256×14×14</TD></TR>
            <TR><TD>layer4 (BasicBlock)</TD><TD ALIGN="RIGHT"><B>512×7×7</B></TD></TR>
            </TABLE>>''', shape='plaintext', fillcolor='white')

        # SPP
        c.node('rgb_spp', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>SPP (空间金字塔池化)</B></TD></TR>
            <TR><TD>金字塔等级: (1, 2, 4)</TD></TR>
            <TR><TD>计算: 512×(1²+2²+4²)</TD></TR>
            <TR><TD><FONT COLOR="blue">Out: 10752</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_rgb_bg, color=color_rgb_border)
        
        # Projection
        c.node('rgb_proj', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>线性投影模块</B></TD></TR>
            <TR><TD>Linear (10752 → 512)</TD></TR>
            <TR><TD><FONT COLOR="blue">rgb_feat (512)</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_rgb_bg, color=color_rgb_border)

        c.edge('rgb_input', 'rgb_backbone')
        c.edge('rgb_backbone', 'rgb_spp')
        c.edge('rgb_spp', 'rgb_proj')

    # --- 右路分支：Dynamic Input ---
    with dot.subgraph(name='cluster_dyn') as c:
        c.attr(label='右路分支 (Dynamic Temporal)', style='dashed', color=color_dyn_border, fontcolor=color_dyn_border)
        
        # 输入
        c.node('dyn_input', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>右路输入</B></TD></TR>
            <TR><TD>动态成像 (帧序列加权)</TD></TR>
            <TR><TD><I>MTCNN 检测与裁剪</I></TD></TR>
            <TR><TD><FONT COLOR="red">3×224×224</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_dyn_bg, color=color_dyn_border)

        # Backbone (Same structure)
        c.node('dyn_backbone', '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" COLOR="#CC6600">
            <TR><TD COLSPAN="2"><B>ResNet-18 (截断至 layer4)</B></TD></TR>
            <TR><TD COLSPAN="2"><I>加载 JAFFE 预训练权重</I></TD></TR>
            <TR><TD>conv1 → bn1 → relu</TD><TD ALIGN="RIGHT">64×112×112</TD></TR>
            <TR><TD>maxpool</TD><TD ALIGN="RIGHT">64×56×56</TD></TR>
            <TR><TD>layer1 (BasicBlock)</TD><TD ALIGN="RIGHT">64×56×56</TD></TR>
            <TR><TD>layer2 (BasicBlock)</TD><TD ALIGN="RIGHT">128×28×28</TD></TR>
            <TR><TD>layer3 (BasicBlock)</TD><TD ALIGN="RIGHT">256×14×14</TD></TR>
            <TR><TD>layer4 (BasicBlock)</TD><TD ALIGN="RIGHT"><B>512×7×7</B></TD></TR>
            </TABLE>>''', shape='plaintext', fillcolor='white')
        
        # SPP
        c.node('dyn_spp', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>SPP (空间金字塔池化)</B></TD></TR>
            <TR><TD>金字塔等级: (1, 2, 4)</TD></TR>
            <TR><TD>计算: 512×(1²+2²+4²)</TD></TR>
            <TR><TD><FONT COLOR="red">Out: 10752</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_dyn_bg, color=color_dyn_border)

        # Projection
        c.node('dyn_proj', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>线性投影模块</B></TD></TR>
            <TR><TD>Linear (10752 → 512)</TD></TR>
            <TR><TD><FONT COLOR="red">dyn_feat (512)</FONT></TD></TR>
            </TABLE>>''', fillcolor=color_dyn_bg, color=color_dyn_border)

        c.edge('dyn_input', 'dyn_backbone')
        c.edge('dyn_backbone', 'dyn_spp')
        c.edge('dyn_spp', 'dyn_proj')

    # --- 融合与分类头 ---
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(style='invis') # 隐形容器用于对齐
        
        # 融合层
        dot.node('concat', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>特征级晚期融合 (Concat)</B></TD></TR>
            <TR><TD>Input: [rgb_feat, dyn_feat]</TD></TR>
            <TR><TD>512 + 512</TD></TR>
            <TR><TD><B>fused (1024)</B></TD></TR>
            </TABLE>>''', shape='octagon', fillcolor=color_shared, style='filled')

        # 分类头
        dot.node('head', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>分类头 (Classification Head)</B></TD></TR>
            <TR><TD>Dropout (p=0.5)</TD></TR>
            <TR><TD>Fully Connected (1024 → 3)</TD></TR>
            </TABLE>>''', fillcolor=color_head)

        # 输出
        dot.node('output', '''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>三分类概率输出</B></TD></TR>
            <TR><TD ALIGN="LEFT">1. Positive (Happiness)</TD></TR>
            <TR><TD ALIGN="LEFT">2. Negative (Disgust/Repression/Sadness/Fear)</TD></TR>
            <TR><TD ALIGN="LEFT">3. Surprise (Surprise)</TD></TR>
            </TABLE>>''', shape='note', fillcolor='#FFFFCC')

    # 连接融合部分
    dot.edge('rgb_proj', 'concat')
    dot.edge('dyn_proj', 'concat')
    dot.edge('concat', 'head')
    dot.edge('head', 'output')

    # --- 训练策略与说明 (Side Note) ---
    note_html = '''<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" BGCOLOR="#F9F9F9">
    <TR><TD ALIGN="LEFT"><B>训练策略与参数配置：</B></TD></TR>
    <TR><TD ALIGN="LEFT">• <B>Loss:</B> FocalLoss (gamma=2) 动态类权重</TD></TR>
    <TR><TD ALIGN="LEFT">• <B>Optimizer:</B> AdamW (lr=5e-4, weight_decay=1e-4)</TD></TR>
    <TR><TD ALIGN="LEFT">• <B>Scheduler:</B> Warmup(8) + CosineAnnealing</TD></TR>
    <TR><TD ALIGN="LEFT">• <B>Control:</B> clip_grad_norm=1.0, EarlyStopping(p=30)</TD></TR>
    <TR><TD ALIGN="LEFT">• <B>Validation:</B> LOSO (Leave-One-Subject-Out)</TD></TR>
    </TABLE>>'''
    
    dot.node('notes', note_html, shape='component', width='3')
    
    # 将说明放在图的底部或侧边（通过不可见边连接）
    dot.edge('output', 'notes', style='invis')

    # 渲染
    output_path = 'ME_Network_Structure'
    dot.render(output_path, view=False, cleanup=False)
    print(f"图表已生成: {output_path}.pdf 及 {output_path}.png (需手动转换)")

if __name__ == '__main__':
    try:
        create_micro_expression_architecture()
    except Exception as e:
        print("运行出错，请检查是否安装了 Graphviz 软件及 python-graphviz 库。")
        print(f"错误详情: {e}")