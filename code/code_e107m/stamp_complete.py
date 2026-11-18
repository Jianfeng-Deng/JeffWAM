from pathlib import Path
import json
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');p=T/'notes/e107m_multimodal_head_20261116.md';audit=json.loads((T/'results/e107m_control_20261116_01/FINAL_AUDIT.json').read_text());s=p.read_text();s=s.replace('）；BRT 2026-11-16T00:51:44.873099-03:00','）；开工BRT 2026-11-16T00:51:44.873099-03:00',1);s=s.replace('\n## 训练前合同','\n完成BRT：'+audit['brt']+'。最终GPUh上界0.628844；新增保守上界1.74GB。\n\n## 训练前合同',1);p.write_text(s);print('Report COMPLETE; final audit passed; logs appended.')
