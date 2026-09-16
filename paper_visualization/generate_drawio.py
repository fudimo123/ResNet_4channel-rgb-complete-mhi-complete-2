import base64
import os
import xml.etree.ElementTree as ET

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return ""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def create_mx_cell(id, value, style, vertex, parent, x, y, width, height):
    cell = ET.Element('mxCell')
    cell.set('id', str(id))
    cell.set('value', value)
    cell.set('style', style)
    cell.set('vertex', str(vertex))
    cell.set('parent', str(parent))
    
    geometry = ET.SubElement(cell, 'mxGeometry')
    geometry.set('x', str(x))
    geometry.set('y', str(y))
    geometry.set('width', str(width))
    geometry.set('height', str(height))
    geometry.set('as', 'geometry')
    
    return cell

def create_edge(id, source, target, parent):
    cell = ET.Element('mxCell')
    cell.set('id', str(id))
    cell.set('style', "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;")
    cell.set('edge', "1")
    cell.set('parent', str(parent))
    cell.set('source', str(source))
    cell.set('target', str(target))
    
    geometry = ET.SubElement(cell, 'mxGeometry')
    geometry.set('relative', "1")
    geometry.set('as', 'geometry')
    
    return cell

def generate_drawio_xml():
    # Paths
    static_img_path = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub01\EP02_01f\img46.jpg"
    dynamic_img_path = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class copy\dynamic_data\sub01\s1_EP02_01f.jpg"
    
    # XML Root
    mxfile = ET.Element('mxfile')
    mxfile.set('host', 'Electron')
    diagram = ET.SubElement(mxfile, 'diagram')
    diagram.set('id', 'FusionNetwork')
    diagram.set('name', 'Fusion Network Architecture')
    
    mxGraphModel = ET.SubElement(diagram, 'mxGraphModel')
    mxGraphModel.set('dx', '1422')
    mxGraphModel.set('dy', '794')
    mxGraphModel.set('grid', '1')
    mxGraphModel.set('gridSize', '10')
    mxGraphModel.set('guides', '1')
    mxGraphModel.set('tooltips', '1')
    mxGraphModel.set('connect', '1')
    mxGraphModel.set('arrows', '1')
    mxGraphModel.set('fold', '1')
    mxGraphModel.set('page', '1')
    mxGraphModel.set('pageScale', '1')
    mxGraphModel.set('pageWidth', '850')
    mxGraphModel.set('pageHeight', '1100')
    mxGraphModel.set('background', '#ffffff')
    
    root = ET.SubElement(mxGraphModel, 'root')
    
    # Default Layers
    ET.SubElement(root, 'mxCell', {'id': '0'})
    ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})
    
    # Styles
    # 3D Cube Style
    style_cube = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    style_process = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#fff2cc;strokeColor=#d6b656;"
    style_img = "shape=image;html=1;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image="
    style_concat = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;fillColor=#e1d5e7;strokeColor=#9673a6;"
    style_fc = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#d5e8d4;strokeColor=#82b366;"
    
    cell_id = 2
    x_start = 50
    y_static = 100
    y_dynamic = 400
    block_width = 100
    block_height = 60
    gap = 140
    
    # --- Static Stream (RGB) ---
    
    # 1. Input Image
    img_data = get_base64_image(static_img_path)
    static_input_id = cell_id
    root.append(create_mx_cell(cell_id, "Static Input", style_img + img_data, 1, 1, x_start, y_static, 100, 100))
    cell_id += 1
    
    current_x = x_start + 150
    prev_id = static_input_id
    
    # ResNet Layers
    resnet_layers = ["Conv1", "Layer1", "Layer2", "Layer3", "Layer4"]
    static_layer_ids = []
    
    for layer in resnet_layers:
        layer_id = cell_id
        root.append(create_mx_cell(layer_id, f"ResNet-18\n{layer}", style_cube, 1, 1, current_x, y_static + 20, block_width, block_height))
        # Edge from previous
        edge = create_edge(cell_id+1000, prev_id, layer_id, 1)
        root.append(edge)
        
        static_layer_ids.append(layer_id)
        prev_id = layer_id
        cell_id += 1
        current_x += gap
        
    # CBAM
    cbam_id = cell_id
    root.append(create_mx_cell(cbam_id, "CBAM", style_process, 1, 1, current_x, y_static + 20, block_width, block_height))
    root.append(create_edge(cell_id+1000, prev_id, cbam_id, 1))
    prev_id = cbam_id
    cell_id += 1
    current_x += gap
    
    # Grid Pooling
    grid_id = cell_id
    root.append(create_mx_cell(grid_id, "Grid Pool\n(2x2)", style_cube, 1, 1, current_x, y_static + 20, block_width, block_height))
    root.append(create_edge(cell_id+1000, prev_id, grid_id, 1))
    prev_id = grid_id
    cell_id += 1
    
    last_static_id = grid_id
    final_x_static = current_x
    
    # --- Dynamic Stream (Optical Flow) ---
    
    # 1. Input Image
    img_data_dyn = get_base64_image(dynamic_img_path)
    dynamic_input_id = cell_id
    root.append(create_mx_cell(cell_id, "Dynamic Input", style_img + img_data_dyn, 1, 1, x_start, y_dynamic, 100, 100))
    cell_id += 1
    
    current_x = x_start + 150
    prev_id = dynamic_input_id
    
    # ResNet Layers
    dynamic_layer_ids = []
    for layer in resnet_layers:
        layer_id = cell_id
        root.append(create_mx_cell(layer_id, f"ResNet-18\n{layer}", style_cube, 1, 1, current_x, y_dynamic + 20, block_width, block_height))
        root.append(create_edge(cell_id+1000, prev_id, layer_id, 1))
        dynamic_layer_ids.append(layer_id)
        prev_id = layer_id
        cell_id += 1
        current_x += gap
        
    # SPP
    spp_id = cell_id
    root.append(create_mx_cell(spp_id, "SPP\n(1x1, 2x2, 4x4)", style_cube, 1, 1, current_x, y_dynamic + 20, block_width, block_height))
    root.append(create_edge(cell_id+1000, prev_id, spp_id, 1))
    prev_id = spp_id
    cell_id += 1
    
    last_dynamic_id = spp_id
    final_x_dynamic = current_x
    
    # --- Fusion ---
    
    fusion_x = max(final_x_static, final_x_dynamic) + 100
    fusion_y = (y_static + y_dynamic) / 2 + 50
    
    # Concatenation
    concat_id = cell_id
    root.append(create_mx_cell(concat_id, "Concat", style_concat, 1, 1, fusion_x, fusion_y, 60, 80))
    cell_id += 1
    
    # Edges to Concat
    # Static to Concat
    edge_static = create_edge(cell_id+1000, last_static_id, concat_id, 1)
    # Adjust exit point for static
    root.append(edge_static)
    
    # Dynamic to Concat
    edge_dynamic = create_edge(cell_id+1001, last_dynamic_id, concat_id, 1)
    root.append(edge_dynamic)
    
    current_x = fusion_x + 100
    prev_id = concat_id
    
    # SE Block
    se_id = cell_id
    root.append(create_mx_cell(se_id, "SE Block\n(No GAP)", style_process, 1, 1, current_x, fusion_y + 10, block_width, 60))
    root.append(create_edge(cell_id+1002, prev_id, se_id, 1))
    prev_id = se_id
    cell_id += 1
    current_x += gap
    
    # Classification
    fc_id = cell_id
    root.append(create_mx_cell(fc_id, "Classification\n(FC)", style_fc, 1, 1, current_x, fusion_y + 10, 80, 80))
    root.append(create_edge(cell_id+1003, prev_id, fc_id, 1))
    
    # Output to File
    # Use absolute path to ensure we write to the correct location
    output_dir = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "fusion_network_structure_v2.xml")
        
    tree = ET.ElementTree(mxfile)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_drawio_xml()
