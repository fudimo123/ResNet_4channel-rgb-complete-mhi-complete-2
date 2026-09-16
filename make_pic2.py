from graphviz import Digraph
import os

# 如果您之前配置过环境变量，这里可能不需要；如果报错找不到 dot，请取消下一行的注释并修改路径
# os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

def create_dynamic_image_flowchart():
    # 初始化图表
    dot = Digraph(comment='Dynamic_Image_Extraction_Flow', format='pdf')
    dot.attr(rankdir='TB', splines='ortho', dpi='300', charset='UTF-8')
    
    # 字体与节点默认设置
    font_name = 'SimHei'  # Windows下常用黑体，Mac可用 'PingFang SC'
    dot.attr('node', shape='box', style='filled', fontname=font_name, fontsize='10', margin='0.15')
    dot.attr('edge', fontname=font_name, fontsize='9')
    dot.attr('graph', fontname=font_name, labelloc='t', fontsize='14')

    # 标题
    dot.attr(label=r'图标题：微表情片段的动态成像提取流程图\n\n')

    # 颜色定义
    c_process = '#E6F3FF'   # 浅蓝：处理步骤
    c_decision = '#FFF0E6'  # 浅橙：判定
    c_io = '#E8F6F3'        # 浅绿：输入输出
    c_note = '#FFFFCC'      # 浅黄：注释/说明
    c_border = '#333333'

    # --- 1. 输入与序列加载 ---
    dot.node('start', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>帧序列加载</B></TD></TR>
        <TR><TD>读取片段目录 &amp; 按帧号排序</TD></TR>
        <TR><TD><FONT COLOR="blue">Input: [img1, img2, ..., imgN]</FONT></TD></TR>
        <TR><TD>Single Frame: 3×H×W</TD></TR>
        </TABLE>>''', shape='box', fillcolor=c_io, color=c_border)

    # --- 2. 有效性判断 (菱形) ---
    dot.node('decision', 'N > 0 ?\n(是否存在有效帧)', shape='diamond', fillcolor=c_decision, color=c_border, height='1.2')
    
    # 异常分支
    dot.node('skip', '返回空 / 跳过该片段', shape='box', style='dashed,filled', fillcolor='#F2F2F2', fontcolor='#666666')

    # --- 3. 权重计算 ---
    dot.node('calc_coeff', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>时间权重系数计算</B></TD></TR>
        <TR><TD ALIGN="LEFT">1. 逐帧计算 (i=1..N):</TD></TR>
        <TR><TD><I>coeff[i] = Σ(j=i..N-1) [(2*(j+1)-N-1) / (j+1)]</I></TD></TR>
        <TR><TD ALIGN="LEFT">2. L1 归一化:</TD></TR>
        <TR><TD><I>coefficients /= sum(|coefficients|)</I></TD></TR>
        <TR><TD><FONT COLOR="blue">Output: Vector [N]</FONT></TD></TR>
        </TABLE>>''', fillcolor=c_process, color=c_border)

    # --- 4. 加权汇聚 ---
    dot.node('aggregation', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>加权汇聚生成动态图</B></TD></TR>
        <TR><TD ALIGN="LEFT">1. Init: dynamic_image = zeros_like(img1)</TD></TR>
        <TR><TD ALIGN="LEFT">2. Loop (i=1..N):</TD></TR>
        <TR><TD><I>dynamic_image += coefficients[i] * frame[i]</I></TD></TR>
        <TR><TD><FONT COLOR="blue">Dim: 3×H×W (float32)</FONT></TD></TR>
        </TABLE>>''', fillcolor=c_process, color=c_border)

    # --- 5. 归一化与转换 ---
    dot.node('normalization', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>归一化与类型转换</B></TD></TR>
        <TR><TD>Min-Max Normalize → [0, 255]</TD></TR>
        <TR><TD>Cast to uint8</TD></TR>
        </TABLE>>''', fillcolor=c_process, color=c_border)

    # --- 6. 命名与保存 ---
    dot.node('save', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>命名与保存 (Output)</B></TD></TR>
        <TR><TD ALIGN="LEFT">CASME2 规范:</TD></TR>
        <TR><TD><I>dynamic_data/subXX/s{subject}_{sequence}.jpg</I></TD></TR>
        <TR><TD><FONT COLOR="blue">Final: 3×H×W (uint8)</FONT></TD></TR>
        </TABLE>>''', shape='folder', fillcolor=c_io, color=c_border)

    # --- 7. 数据集对接说明 (右侧注释) ---
    dot.node('dataset_note', '''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>数据集对接说明</B></TD></TR>
        <TR><TD ALIGN="LEFT">训练阶段加载时：</TD></TR>
        <TR><TD ALIGN="LEFT">1. MTCNN 人脸检测与裁剪</TD></TR>
        <TR><TD ALIGN="LEFT">2. 缩放至统一尺寸</TD></TR>
        <TR><TD><B>Input Size: 3×224×224</B></TD></TR>
        </TABLE>>''', shape='note', fillcolor=c_note, width='2.5')

    # --- 构建连线 ---
    dot.edge('start', 'decision')
    
    # 分支连线
    dot.edge('decision', 'skip', label=' No (N=0)')
    dot.edge('decision', 'calc_coeff', label=' Yes')
    
    # 主流程连线
    dot.edge('calc_coeff', 'aggregation')
    dot.edge('aggregation', 'normalization')
    dot.edge('normalization', 'save')

    # 注释连线 (虚线)
    dot.edge('save', 'dataset_note', style='dotted', arrowhead='none', constraint='false')

    # 渲染
    output_path = 'Dynamic_Image_Flowchart'
    dot.render(output_path, view=False, cleanup=False)
    print(f"图表已生成: {output_path}.pdf 及 {output_path}.png (需手动转换)")

if __name__ == '__main__':
    try:
        create_dynamic_image_flowchart()
    except Exception as e:
        print(f"运行出错: {e}")