from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');root=T/'code_e107m'
for name in ('e99w2_common.py','e103d_runtime.py'):
 p=root/name;s=p.read_text().replace("code_e106c","code_e107m").replace("e106c_control_20261115_01","e107m_control_20261116_01").replace("e106c_longer_training_20261115.md","e107m_multimodal_head_20261116.md").replace("e106c_f2_s90_20261115_01","e107m_m0_s90_20261116_01")
 p.write_text(s)
