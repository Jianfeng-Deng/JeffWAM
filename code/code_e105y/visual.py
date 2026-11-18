from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=Path((T/'code_e105y/run_path.txt').read_text().strip());p=json.loads((T/'data/e103d_cache/plan.json').read_text());rs=[r for r in p['records'] if r['source']=='yam'];sel=[rs[i] for i in [0,1,14,28,29,35,42]];im=Image.new('RGB',(640,7*180),(255,255,255));d=ImageDraw.Draw(im)
for i,r in enumerate(sel):
 raw=Path(r['original']['representative_path'])
 for j,k in enumerate(['top','left','right']):
  cap=cv2.VideoCapture(str(raw/(k+'-images-rgb.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,30);ok,f=cap.read();cap.release();assert ok
  v=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));v.thumbnail((210,155));im.paste(v,(j*213,i*180+23))
 d.text((3,i*180+3),r['canonical_id'],fill='black')
f=R/'camera_contact.jpg';assert not f.exists();im.save(f,quality=85);print(f)
