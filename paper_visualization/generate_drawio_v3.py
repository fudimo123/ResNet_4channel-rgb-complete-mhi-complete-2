import base64
import os
import xml.etree.ElementTree as ET

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return ""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Debug: print first few chars
        print(f"Encoded {image_path}: {len(encoded_string)} chars, starts with {encoded_string[:20]}...")
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

def create_edge(id, source, target, parent, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"):
    cell = ET.Element('mxCell')
    cell.set('id', str(id))
    cell.set('style', style)
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
    diagram.set('id', 'FusionNetworkV3')
    diagram.set('name', 'Fusion Network Final')
    
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
    mxGraphModel.set('pageWidth', '1654')
    mxGraphModel.set('pageHeight', '1169')
    mxGraphModel.set('background', '#ffffff')
    
    root = ET.SubElement(mxGraphModel, 'root')
    
    # Default Layers
    ET.SubElement(root, 'mxCell', {'id': '0'})
    ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})
    
    # --- Styles (Reverting to V1 look for modules, New look for Backbone) ---
    
    # Input Images
    style_img = "shape=image;html=1;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image="
    
    # Backbone Group (Dashed Box)
    style_backbone_group_rgb = "group;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#6c8ebf;verticalAlign=top;align=center;spacingTop=5;fontStyle=1;fontSize=14;"
    style_backbone_group_dyn = "group;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#d79b00;verticalAlign=top;align=center;spacingTop=5;fontStyle=1;fontSize=14;"
    
    # Backbone Layers (Cubes inside the group)
    style_layer_rgb = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    style_layer_dyn = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#ffe6cc;strokeColor=#d79b00;"
    
    # Feature Maps (Cubes)
    style_feat_rgb = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    style_feat_dyn = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#ffe6cc;strokeColor=#d79b00;"
    
    # Modules (Reverted to V1: Rounded Rects)
    style_cbam = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontStyle=1"
    style_grid = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontStyle=1"
    style_spp = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontStyle=1"
    
    # Vectors (Rects)
    style_vec_rgb = "rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    style_vec_dyn = "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;"
    style_proj = "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;"
    
    # Fusion (Process Shape from V1)
    style_concat = "shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1"
    style_fused_vec = "rounded=0;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"
    
    # SE Block Group (Swimlane from V1)
    style_se_group = "swimlane;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;dashed=1;"
    style_se_inner = "rounded=1;whiteSpace=wrap;html=1;"
    style_se_op = "ellipse;whiteSpace=wrap;html=1;"
    
    # Output
    style_final_fc = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontStyle=1"
    style_out_pos = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#d5e8d4;strokeColor=#82b366;"
    style_out_neg = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#f8cecc;strokeColor=#b85450;"
    style_out_sur = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fff2cc;strokeColor=#d6b656;"
    
    cell_id = 100
    
    # Layout Constants
    x_start = 40
    y_rgb = 160
    y_dyn = 500
    
    # ---------------- RGB STREAM ----------------
    
    # 1. Image
    img_data_rgb = get_base64_image(static_img_path)
    rgb_input_id = cell_id
    root.append(create_mx_cell(rgb_input_id, "Static Input", style_img + img_data_rgb, 1, 1, x_start, y_rgb, 100, 100))
    cell_id += 1
    
    # 2. Backbone Group
    bb_x = x_start + 150
    bb_w = 420
    bb_h = 140
    rgb_bb_id = cell_id
    # Note: Use container=1 for groups? In XML, simple vertex with children pointing to it works as group if we treat coordinates relatively.
    # However, simple way: Just draw the dashed box behind (parent=1) and draw items inside (parent=1) on top.
    # But user wants "one big package". A Group cell is best.
    # Let's try creating the group cell.
    root.append(create_mx_cell(rgb_bb_id, "ResNet-18 Backbone", style_backbone_group_rgb, 1, 1, bb_x, y_rgb - 20, bb_w, bb_h))
    cell_id += 1
    
    # Edge Input -> Backbone
    root.append(create_edge(cell_id, rgb_input_id, rgb_bb_id, 1))
    cell_id += 1
    
    # Layers inside Backbone (Relative coordinates if parent is group, but let's use absolute for safety if not using swimlane)
    # Actually, if I set parent=rgb_bb_id, coordinates are relative to group.
    layer_names = ["Conv1", "Layer1", "Layer2", "Layer3", "Layer4"]
    prev_layer_id = None
    lx = 20
    ly = 40
    l_gap = 80
    
    rgb_last_layer_id = None
    
    for layer in layer_names:
        lid = cell_id
        root.append(create_mx_cell(lid, layer, style_layer_rgb, 1, rgb_bb_id, lx, ly, 60, 50))
        if prev_layer_id:
            # Edge inside group
            edge_id = cell_id + 1000
            root.append(create_edge(edge_id, prev_layer_id, lid, rgb_bb_id))
        prev_layer_id = lid
        rgb_last_layer_id = lid
        cell_id += 1
        lx += l_gap
        
    # 3. Feature Map (Output of Layer4)
    feat_x = bb_x + bb_w + 50
    rgb_feat_id = cell_id
    root.append(create_mx_cell(rgb_feat_id, "512x7x7", style_feat_rgb, 1, 1, feat_x, y_rgb + 20, 60, 60))
    # Edge from Backbone Group to Feature
    root.append(create_edge(cell_id+1, rgb_bb_id, rgb_feat_id, 1))
    cell_id += 2
    
    # 4. CBAM
    cbam_x = feat_x + 100
    cbam_id = cell_id
    root.append(create_mx_cell(cbam_id, "CBAM", style_cbam, 1, 1, cbam_x, y_rgb + 20, 100, 60))
    root.append(create_edge(cell_id+1, rgb_feat_id, cbam_id, 1))
    cell_id += 2
    
    # 5. Grid Pool
    grid_x = cbam_x + 140
    grid_id = cell_id
    root.append(create_mx_cell(grid_id, "Grid Pool\n(2x2)", style_grid, 1, 1, grid_x, y_rgb + 20, 100, 60))
    root.append(create_edge(cell_id+1, cbam_id, grid_id, 1))
    cell_id += 2
    
    # 6. Vectors
    vec_x = grid_x + 140
    rgb_vec_id = cell_id
    root.append(create_mx_cell(rgb_vec_id, "2048-dim", style_vec_rgb, 1, 1, vec_x, y_rgb + 35, 80, 30))
    root.append(create_edge(cell_id+1, grid_id, rgb_vec_id, 1))
    cell_id += 2
    
    proj_x = vec_x + 110
    rgb_proj_id = cell_id
    root.append(create_mx_cell(rgb_proj_id, "Linear+ReLU", style_proj, 1, 1, proj_x, y_rgb + 35, 80, 30))
    root.append(create_edge(cell_id+1, rgb_vec_id, rgb_proj_id, 1))
    cell_id += 2
    
    final_x = proj_x + 110
    rgb_final_id = cell_id
    root.append(create_mx_cell(rgb_final_id, "512-dim", style_vec_rgb, 1, 1, final_x, y_rgb + 35, 80, 30))
    root.append(create_edge(cell_id+1, rgb_proj_id, rgb_final_id, 1))
    cell_id += 2
    
    last_rgb_x = final_x
    
    # ---------------- DYNAMIC STREAM ----------------
    
    # 1. Image
    img_data_dyn = get_base64_image(dynamic_img_path)
    dyn_input_id = cell_id
    root.append(create_mx_cell(dyn_input_id, "Dynamic Input", style_img + img_data_dyn, 1, 1, x_start, y_dyn, 100, 100))
    cell_id += 1
    
    # 2. Backbone Group
    dyn_bb_id = cell_id
    root.append(create_mx_cell(dyn_bb_id, "ResNet-18 Backbone", style_backbone_group_dyn, 1, 1, bb_x, y_dyn - 20, bb_w, bb_h))
    cell_id += 1
    
    root.append(create_edge(cell_id, dyn_input_id, dyn_bb_id, 1))
    cell_id += 1
    
    # Layers
    prev_layer_id = None
    lx = 20
    for layer in layer_names:
        lid = cell_id
        root.append(create_mx_cell(lid, layer, style_layer_dyn, 1, dyn_bb_id, lx, ly, 60, 50))
        if prev_layer_id:
            edge_id = cell_id + 1000
            root.append(create_edge(edge_id, prev_layer_id, lid, dyn_bb_id))
        prev_layer_id = lid
        cell_id += 1
        lx += l_gap
        
    # 3. Feature Map
    dyn_feat_id = cell_id
    root.append(create_mx_cell(dyn_feat_id, "512x7x7", style_feat_dyn, 1, 1, feat_x, y_dyn + 20, 60, 60))
    root.append(create_edge(cell_id+1, dyn_bb_id, dyn_feat_id, 1))
    cell_id += 2
    
    # 4. SPP
    spp_x = feat_x + 100
    spp_id = cell_id
    root.append(create_mx_cell(spp_id, "SPP\n(1x1, 2x2, 4x4)", style_spp, 1, 1, spp_x, y_dyn + 20, 100, 60))
    root.append(create_edge(cell_id+1, dyn_feat_id, spp_id, 1))
    cell_id += 2
    
    # 5. Vectors
    vec_x = spp_x + 140
    dyn_vec_id = cell_id
    root.append(create_mx_cell(dyn_vec_id, "10752-dim", style_vec_dyn, 1, 1, vec_x, y_dyn + 35, 100, 30)) # Wider for 10752
    root.append(create_edge(cell_id+1, spp_id, dyn_vec_id, 1))
    cell_id += 2
    
    proj_x = vec_x + 140
    dyn_proj_id = cell_id
    root.append(create_mx_cell(dyn_proj_id, "Linear+ReLU", style_proj, 1, 1, proj_x, y_dyn + 35, 80, 30))
    root.append(create_edge(cell_id+1, dyn_vec_id, dyn_proj_id, 1))
    cell_id += 2
    
    final_x = proj_x + 110
    dyn_final_id = cell_id
    root.append(create_mx_cell(dyn_final_id, "512-dim", style_vec_dyn, 1, 1, final_x, y_dyn + 35, 80, 30))
    root.append(create_edge(cell_id+1, dyn_proj_id, dyn_final_id, 1))
    cell_id += 2
    
    # ---------------- FUSION ----------------
    
    fusion_x = last_rgb_x + 100
    fusion_y = (y_rgb + y_dyn) / 2
    
    # Concat
    concat_id = cell_id
    root.append(create_mx_cell(concat_id, "Concat", style_concat, 1, 1, fusion_x, fusion_y, 80, 40))
    cell_id += 1
    
    # Edges to Concat
    root.append(create_edge(cell_id, rgb_final_id, concat_id, 1))
    cell_id += 1
    root.append(create_edge(cell_id, dyn_final_id, concat_id, 1))
    cell_id += 1
    
    # Fused Vector
    fused_vec_id = cell_id
    root.append(create_mx_cell(fused_vec_id, "1024-dim", style_fused_vec, 1, 1, fusion_x + 100, fusion_y + 5, 80, 30))
    root.append(create_edge(cell_id+1, concat_id, fused_vec_id, 1))
    cell_id += 2
    
    # ---------------- SE BLOCK (Detailed) ----------------
    
    se_x = fusion_x + 220
    se_y = fusion_y + 50
    se_w = 300
    se_h = 100
    
    se_group_id = cell_id
    root.append(create_mx_cell(se_group_id, "SE-Block (No GAP)", style_se_group, 1, 1, se_x, se_y, se_w, se_h))
    cell_id += 1
    
    # Edge from Fused to SE Group (just visual, actually to first FC)
    # But first we need children
    
    se_fc1_id = cell_id
    root.append(create_mx_cell(se_fc1_id, "FC(64)", style_se_inner, 1, se_group_id, 20, 35, 60, 30))
    cell_id += 1
    
    se_relu_id = cell_id
    root.append(create_mx_cell(se_relu_id, "ReLU", style_se_op, 1, se_group_id, 90, 35, 40, 30))
    cell_id += 1
    
    se_fc2_id = cell_id
    root.append(create_mx_cell(se_fc2_id, "FC(1024)", style_se_inner, 1, se_group_id, 140, 35, 60, 30))
    cell_id += 1
    
    se_sig_id = cell_id
    root.append(create_mx_cell(se_sig_id, "Sigmoid", style_se_op, 1, se_group_id, 210, 35, 50, 30))
    cell_id += 1
    
    # Internal SE Edges
    root.append(create_edge(cell_id, se_fc1_id, se_relu_id, se_group_id))
    cell_id += 1
    root.append(create_edge(cell_id, se_relu_id, se_fc2_id, se_group_id))
    cell_id += 1
    root.append(create_edge(cell_id, se_fc2_id, se_sig_id, se_group_id))
    cell_id += 1
    
    # Edge Fused -> SE FC1
    root.append(create_edge(cell_id, fused_vec_id, se_fc1_id, 1))
    cell_id += 1
    
    # Multiply Node
    mult_id = cell_id
    mult_x = se_x + se_w + 40
    root.append(create_mx_cell(mult_id, "⊗", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontSize=20;fillColor=#d5e8d4;strokeColor=#82b366;", 1, 1, mult_x, fusion_y, 40, 40))
    cell_id += 1
    
    # Edge Fused -> Mult
    root.append(create_edge(cell_id, fused_vec_id, mult_id, 1))
    cell_id += 1
    
    # Edge Sigmoid -> Mult
    root.append(create_edge(cell_id, se_sig_id, mult_id, 1))
    cell_id += 1
    
    # ---------------- OUTPUT ----------------
    
    final_fc_id = cell_id
    root.append(create_mx_cell(final_fc_id, "FC + Dropout", style_final_fc, 1, 1, mult_x + 80, fusion_y, 100, 40))
    root.append(create_edge(cell_id+1, mult_id, final_fc_id, 1))
    cell_id += 2
    
    # Classes
    out_x = mult_x + 220
    
    pos_id = cell_id
    root.append(create_mx_cell(pos_id, "Pos", style_out_pos, 1, 1, out_x, fusion_y - 40, 40, 40))
    cell_id += 1
    
    neg_id = cell_id
    root.append(create_mx_cell(neg_id, "Neg", style_out_neg, 1, 1, out_x, fusion_y, 40, 40))
    cell_id += 1
    
    sur_id = cell_id
    root.append(create_mx_cell(sur_id, "Sur", style_out_sur, 1, 1, out_x, fusion_y + 40, 40, 40))
    cell_id += 1
    
    root.append(create_edge(cell_id, final_fc_id, pos_id, 1))
    cell_id += 1
    root.append(create_edge(cell_id, final_fc_id, neg_id, 1))
    cell_id += 1
    root.append(create_edge(cell_id, final_fc_id, sur_id, 1))
    cell_id += 1
    
    # Output to File
    output_dir = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "fusion_network_structure_v3.xml")
        
    tree = ET.ElementTree(mxfile)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_drawio_xml()
