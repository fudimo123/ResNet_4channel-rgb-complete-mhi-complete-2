import base64
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import io
import os

IMAGE_PATH = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub01\EP02_01f\img46.jpg"
OUTPUT_XML = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\Dynamic_Image_Rank_Pooling.xml"

def img_to_b64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 1. Load Image and create a face crop
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    raw_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
else:
    raw_img = Image.open(IMAGE_PATH).convert('RGB')

width, height = raw_img.size
left, top, right, bottom = width * 0.2, height * 0.1, width * 0.8, height * 0.9
face_img = raw_img.crop((left, top, right, bottom)).resize((120, 120))
b64_face = img_to_b64(face_img)

# 2. Create an embossed/gray version to simulate the "Dynamic Image" result in the user's reference
gray_img = face_img.convert('L')
emboss_img = gray_img.filter(ImageFilter.EMBOSS)
# Convert to RGB so draw.io handles it normally
di_img = emboss_img.convert('RGB')
b64_di = img_to_b64(di_img)

# 3. XML Template mimicking the exact reference diagram
xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" modified="2023-10-27T00:00:00.000Z" agent="Mozilla/5.0" version="21.2.8" type="device">
  <diagram id="dynamic_image_rank_pooling" name="Rank Pooling">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Image Sequence (Stacked) -->
        <mxCell id="seq_bg2" value="" style="shape=image;image=data:image/png;base64,{b64_face};opacity=50;" vertex="1" parent="1">
          <mxGeometry x="80" y="260" width="120" height="120" as="geometry" />
        </mxCell>
        <mxCell id="seq_bg1" value="" style="shape=image;image=data:image/png;base64,{b64_face};opacity=75;" vertex="1" parent="1">
          <mxGeometry x="70" y="270" width="120" height="120" as="geometry" />
        </mxCell>
        <mxCell id="seq_fg" value="" style="shape=image;image=data:image/png;base64,{b64_face};" vertex="1" parent="1">
          <mxGeometry x="60" y="280" width="120" height="120" as="geometry" />
        </mxCell>
        <mxCell id="seq_shadow" value="" style="shape=rect;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;strokeWidth=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="60" y="280" width="120" height="120" as="geometry" />
        </mxCell>
        
        <mxCell id="seq_text" value="Image Sequence&lt;br&gt;$$I_1, I_2, \dots, I_N$$" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="40" y="420" width="160" height="40" as="geometry" />
        </mxCell>

        <!-- Arrow 1 -->
        <mxCell id="arr1" style="edgeStyle=straightEdgeStyle;rounded=0;html=1;strokeWidth=1.5;endArrow=classic;strokeColor=#000000;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="190" y="340" as="sourcePoint" /><mxPoint x="320" y="340" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Calculate Weights Box -->
        <!-- We use a 3D box (cube) to match the reference style -->
        <mxCell id="box_weights" value="Calculate&lt;br&gt;Weights $$C_t$$" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#6c8ebf;strokeColor=#4d6b9c;fontColor=#000000;shadow=1;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="320" y="290" width="140" height="100" as="geometry" />
        </mxCell>
        
        <!-- Formula above Calculate Weights -->
        <mxCell id="formula_ct" value="$$C_t = \sum_{{j=t}}^N \frac{{2(j+1)-N-1}}{{j+1}}$$" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#666666;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="300" y="190" width="180" height="40" as="geometry" />
        </mxCell>
        <!-- Dashed line connecting formula to box -->
        <mxCell id="dash_formula" style="edgeStyle=straightEdgeStyle;rounded=0;html=1;strokeWidth=1;endArrow=none;dashed=1;strokeColor=#cccccc;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="390" y="230" as="sourcePoint" /><mxPoint x="390" y="290" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Arrow 2 -->
        <mxCell id="arr2" style="edgeStyle=straightEdgeStyle;rounded=0;html=1;strokeWidth=1.5;endArrow=classic;strokeColor=#000000;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="460" y="340" as="sourcePoint" /><mxPoint x="530" y="340" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Sigma (Summation) Circle -->
        <mxCell id="circle_sum" value="$$\sum$$" style="shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffebcc;strokeColor=#d6b656;fontColor=#000000;fontSize=28;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="530" y="300" width="80" height="80" as="geometry" />
        </mxCell>
        
        <mxCell id="sum_text" value="Weighted Sum&lt;br&gt;$$D = \sum C_t I_t$$" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="490" y="400" width="160" height="40" as="geometry" />
        </mxCell>

        <!-- Arrow 3 -->
        <mxCell id="arr3" style="edgeStyle=straightEdgeStyle;rounded=0;html=1;strokeWidth=1.5;endArrow=classic;strokeColor=#000000;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="610" y="340" as="sourcePoint" /><mxPoint x="680" y="340" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Normalize Box -->
        <mxCell id="box_norm" value="Normalize&lt;br&gt;0 ~ 255" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#6c8ebf;strokeColor=#4d6b9c;fontColor=#000000;shadow=1;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="680" y="290" width="140" height="100" as="geometry" />
        </mxCell>

        <!-- Arrow 4 -->
        <mxCell id="arr4" style="edgeStyle=straightEdgeStyle;rounded=0;html=1;strokeWidth=1.5;endArrow=classic;strokeColor=#000000;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="820" y="340" as="sourcePoint" /><mxPoint x="910" y="340" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Dynamic Image Result -->
        <mxCell id="di_result" value="" style="shape=image;image=data:image/png;base64,{b64_di};" vertex="1" parent="1">
          <mxGeometry x="910" y="280" width="120" height="120" as="geometry" />
        </mxCell>
        <mxCell id="di_shadow" value="" style="shape=rect;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;strokeWidth=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="910" y="280" width="120" height="120" as="geometry" />
        </mxCell>
        
        <mxCell id="di_text" value="Dynamic Image" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;fontSize=14;" vertex="1" parent="1">
          <mxGeometry x="890" y="420" width="160" height="30" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

with open(OUTPUT_XML, 'w', encoding='utf-8') as f:
    f.write(xml_template)
print("Successfully generated Dynamic_Image_Rank_Pooling.xml")
