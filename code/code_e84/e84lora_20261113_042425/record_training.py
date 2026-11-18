from common import *
state=json.loads((P/'progress.json').read_text())
if not state.get('training_section_recorded'):
 a={v:json.loads((L/v/'training_result.json').read_text()) for v in ['lora','shuffled']};data=json.loads((L/'READY_training_data.json').read_text())
 append_report('## 双组LoRA训练完成\n\n'+brt()+f' 真实与打乱文本均固定768步、双塔最后4层Q/V rank8，共196608可训练参数；视觉与文本各98304。训练库1483对FK+331对E83、191个train parent，两组同初始化/采样顺序。两塔首步梯度检查通过，未用val挑checkpoint；末步loss真实/打乱='+str([a[v]['logs'][-1]['loss'] for v in a])+f'。adapter路径 `{L}`；未合并、未替换原模型。首个设备放置错误发生在优化步0，已归档，GPU时间计入；未OOM。')
 update(training_section_recorded=True,adapters_complete=True)
print('training_section_recorded')
