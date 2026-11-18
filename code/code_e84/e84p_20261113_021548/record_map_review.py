from common import *
from contract import sha
fig=P/'frozen/figures/pixel_qualitative_fixed_samples.png'
notes={'G002':'Orange map emphasizes parts of visible strap; left/right gripper maps are visually similar and include bright border/background. Coordinates align with source overview.','G027':'Overlapping grippers and noisy identity case; left/right maps visually similar. No annotation revised and low flags remain excluded.','G103':'No trusted gripper points visible; orange map highlights ties and basket simultaneously, illustrating object identity/background ambiguity.','G126':'Native full-FOV map transport aligns visible tie/gripper regions; color map highlights vertical orange segment, LR maps remain similar; noisy GT cannot certify fine tip localization.'}
atomic(P/'map_visual_review.json',{'timestamp_brt':brt(),'actual_image_viewed':str(fig),'sha256':sha(fig),'viewed_in_tool':'view_image full4x4 figure','selection':'four fixed G-report representative IDs, no selection by score; qualitative only','notes':notes,'threshold_or_checkpoint_changed':False,'gt_modified':False})
append_report('## 相似度图实际目视复核 '+brt()+'\n\n已实际查看固定G002/G027/G103/G126的4×4原图/左右夹爪/橙带热图 `'+str(fig)+'`。原图回映位置合理；左右查询热图外观相近，橙带响应同时可落在篮筐/多根带上。此为方法/歧义核对，不替代定量结果，不据此改标注、半径、checkpoint。记录 `'+str(P/'map_visual_review.json')+'`。')
print('VISUAL_MAP_REVIEW_RECORDED')
