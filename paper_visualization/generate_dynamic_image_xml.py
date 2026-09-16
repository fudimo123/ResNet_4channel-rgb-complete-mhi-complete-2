import base64
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import io
import os

IMAGE_PATH = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub01\EP02_01f\img46.jpg"
OUTPUT_XML = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\Dynamic_Image_Pipeline.xml"

def img_to_b64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 1. Load Raw Image
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    raw_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
else:
    raw_img = Image.open(IMAGE_PATH).convert('RGB')

width, height = raw_img.size
left, top, right, bottom = width * 0.2, height * 0.1, width * 0.8, height * 0.9

# 1.1 Raw Image with MTCNN Bounding Box
raw_img_with_box = raw_img.copy()
draw = ImageDraw.Draw(raw_img_with_box)
draw.rectangle([left, top, right, bottom], outline="red", width=int(width*0.02))
raw_img_resized = raw_img_with_box.resize((120, 120))
b64_raw = img_to_b64(raw_img_resized)

# 2. Cropped Image
cropped_img = raw_img.crop((left, top, right, bottom)).resize((120, 120))
b64_crop = img_to_b64(cropped_img)

# 3. Dynamic Image Mock
gray = np.array(cropped_img.convert('L'), dtype=np.float32)
# Apply a mock "motion" effect: highlight edges and add some directional gradient
dy, dx = np.gradient(gray)
magnitude = np.sqrt(dx**2 + dy**2)
# Normalize
magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-5)

# Save with colormap to simulate Rank Pooling energy map
plt.imsave('temp_di.png', magnitude, cmap='jet')
di_img = Image.open('temp_di.png').resize((120, 120)).convert('RGB')
b64_di = img_to_b64(di_img)
if os.path.exists('temp_di.png'):
    os.remove('temp_di.png')

# Construct XML
xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" modified="2023-10-27T00:00:00.000Z" agent="Mozilla/5.0" version="21.2.8" type="device">
  <diagram id="dynamic_image_pipeline" name="Dynamic Image Pipeline">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Stage 1: Raw Sequence -->
        <mxCell id="raw1" value="" style="shape=image;image=data:image/png;base64,{b64_raw};" vertex="1" parent="1"><mxGeometry x="50" y="200" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="raw2" value="" style="shape=image;image=data:image/png;base64,{b64_raw};" vertex="1" parent="1"><mxGeometry x="60" y="210" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="raw3" value="" style="shape=image;image=data:image/png;base64,{b64_raw};" vertex="1" parent="1"><mxGeometry x="70" y="220" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="raw_text" value="&lt;b&gt;Raw Video Sequence&lt;/b&gt;&lt;br&gt;(Length: &lt;i&gt;N&lt;/i&gt; frames)" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="40" y="360" width="160" height="40" as="geometry" />
        </mxCell>

        <!-- Arrow 1 -->
        <mxCell id="arr1" value="&lt;b&gt;MTCNN&lt;/b&gt;&lt;br&gt;Face Alignment" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;endArrow=classic;strokeColor=#666666;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="200" y="270" as="sourcePoint" /><mxPoint x="300" y="270" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Stage 2: Cropped Sequence -->
        <mxCell id="crop1" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="310" y="200" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="crop2" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="320" y="210" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="crop3" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="330" y="220" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="crop_text" value="&lt;b&gt;Aligned Face Sequence&lt;/b&gt;&lt;br&gt;(Length: &lt;i&gt;N&lt;/i&gt; frames)" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="300" y="360" width="160" height="40" as="geometry" />
        </mxCell>

        <!-- Arrow 2 -->
        <mxCell id="arr2" value="&lt;b&gt;TIM&lt;/b&gt;&lt;br&gt;Time Interpolation" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;endArrow=classic;strokeColor=#666666;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="460" y="270" as="sourcePoint" /><mxPoint x="560" y="270" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Stage 3: Normalized Sequence -->
        <mxCell id="norm1" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="570" y="200" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="norm2" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="580" y="210" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="norm3" value="" style="shape=image;image=data:image/png;base64,{b64_crop};" vertex="1" parent="1"><mxGeometry x="590" y="220" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="norm_text" value="&lt;b&gt;Normalized Sequence&lt;/b&gt;&lt;br&gt;(Length: &lt;i&gt;T&lt;/i&gt; = 16 or 32)" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="560" y="360" width="160" height="40" as="geometry" />
        </mxCell>

        <!-- Arrow 3 -->
        <mxCell id="arr3" value="&lt;b&gt;Rank Pooling&lt;/b&gt;&lt;br&gt;(SVR)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;endArrow=classic;strokeColor=#666666;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="720" y="270" as="sourcePoint" /><mxPoint x="820" y="270" as="targetPoint" /></mxGeometry>
        </mxCell>

        <!-- Stage 4: Dynamic Image -->
        <mxCell id="di1" value="" style="shape=image;image=data:image/png;base64,{b64_di};" vertex="1" parent="1"><mxGeometry x="830" y="210" width="120" height="120" as="geometry" /></mxCell>
        <mxCell id="di_text" value="&lt;b&gt;Dynamic Image&lt;/b&gt;&lt;br&gt;(1 RGB Image)" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="810" y="360" width="160" height="40" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

with open(OUTPUT_XML, 'w', encoding='utf-8') as f:
    f.write(xml_template)
print("Successfully generated Dynamic_Image_Pipeline.xml")
