# E98-G 跨任务阶段体系泛化

label_source=model(gpt-6-astra)，非人工。纯文本/CPU。

初稿计数断言发现：116条记录涉及16任务，但18条无cleaned段，实际2,484段来自98条记录/15任务。初稿保留draft_01_count_assertion；此报告为修正后的交付。

## 1. 来源与边界
ABC 116 episode，2,484 cleaned_original_segments，349种原文；全部task_category=A、train标识，记录涉及16任务，其中18个episode没有cleaned段，实际段覆盖98个episode、15任务；混合食品装袋任务5条均无cleaned段，覆盖率记null而非0或100%。E89中B/C候选没有进入本次E91输入，不能声称ABC全类别泛化。yam 49 clip、196条当前版叙述；不重复统计v1_value。只读五份必读文件，SHA256见input_manifest.json。E98允许的20个dev parent模型真值例外本会话未使用；世界头及其他dev/test规则不变。源标注的作者/模型信息保留于只读输入，本会话映射统一标记label_source=model(gpt-6-astra)、非人工。

## 2. 通用层与Piper实例
已将10阶段映射至11个核心原语，21谓词无遗漏绑定到18个可复用核心模板；另有10个跨任务扩展原语、7个扩展模板，共21原语/25模板。原Piper转移守卫与证据逐项保留。grasp_lift、place_release为明确复合原语，不推造内部边界；recover是带失败上下文的控制原语。通用层只规定关系与证据，任务层绑定手臂、对象、部件、孔口、目标形状和AND/OR完成守卫。

S01 reach；S02 grasp_lift；S03 handover；S04 reach→regrasp；S05 align；S06 insert；S07 regrasp→pull；S08 place_release；S09 retreat→check；S10 recover。S04/S07新抓点必须保持同一target身份，不能把静态持物当换抓。P073单独保留不可见机械验证；试管盲孔不能要求P062式“出口穿出”。

## 3. 覆盖检验
统计是看过语料后设计规则的描述性覆盖，不是留出泛化准确率。全量记录保留原坐标与文本，不移动边界，不以完成意图回答yes。ABC各段是动作意图；yam各段是稀疏状态叙述，两者分别报告，不能合成一个准确率。

{
  "abc": {
    "n": 2484,
    "counts": {
      "mapped_core": 1424,
      "mapped_extension": 538,
      "unmapped": 63,
      "partial": 459
    },
    "core_explicit_mapping_rate": 0.573268921095008,
    "extended_explicit_mapping_rate": 0.7898550724637681,
    "partial_or_state_not_counted": 459,
    "unmapped": 63
  },
  "abc_unique": {
    "n": 349,
    "counts": {
      "mapped_core": 178,
      "mapped_extension": 96,
      "unmapped": 5,
      "partial": 70
    },
    "core_explicit_mapping_rate": 0.5100286532951289,
    "extended_explicit_mapping_rate": 0.7851002865329513,
    "partial_or_state_not_counted": 70,
    "unmapped": 5
  },
  "yam": {
    "n": 196,
    "counts": {
      "mapped_core": 58,
      "state_only": 118,
      "mapped_extension": 20
    },
    "core_explicit_mapping_rate": 0.29591836734693877,
    "extended_explicit_mapping_rate": 0.3979591836734694,
    "partial_or_state_not_counted": 118,
    "unmapped": 0
  },
  "yam_clip_any_explicit": 49,
  "yam_clip_full_task": 0,
  "abc_tasks": {
    "connect_and_route_the_hose": {
      "n": 51,
      "counts": {
        "mapped_core": 27,
        "mapped_extension": 7,
        "unmapped": 7,
        "partial": 10
      },
      "core_explicit_mapping_rate": 0.5294117647058824,
      "extended_explicit_mapping_rate": 0.6666666666666666,
      "partial_or_state_not_counted": 10,
      "unmapped": 7
    },
    "place_the_bra_in_the_laundry_bag_and_zip_it_closed": {
      "n": 72,
      "counts": {
        "mapped_extension": 31,
        "mapped_core": 22,
        "partial": 19
      },
      "core_explicit_mapping_rate": 0.3055555555555556,
      "extended_explicit_mapping_rate": 0.7361111111111112,
      "partial_or_state_not_counted": 19,
      "unmapped": 0
    },
    "place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot": {
      "n": 34,
      "counts": {
        "mapped_core": 25,
        "mapped_extension": 3,
        "partial": 6
      },
      "core_explicit_mapping_rate": 0.7352941176470589,
      "extended_explicit_mapping_rate": 0.8235294117647058,
      "partial_or_state_not_counted": 6,
      "unmapped": 0
    },
    "place_the_food_into_the_grocery_bag_and_tie_it": {
      "n": 40,
      "counts": {
        "partial": 13,
        "mapped_core": 26,
        "mapped_extension": 1
      },
      "core_explicit_mapping_rate": 0.65,
      "extended_explicit_mapping_rate": 0.675,
      "partial_or_state_not_counted": 13,
      "unmapped": 0
    },
    "place_the_mixed_food_into_the_grocery_bags_and_tie_them": {
      "n": 0,
      "counts": {},
      "core_explicit_mapping_rate": null,
      "extended_explicit_mapping_rate": null,
      "partial_or_state_not_counted": 0,
      "unmapped": 0
    },
    "roll_the_ties": {
      "n": 129,
      "counts": {
        "mapped_core": 61,
        "mapped_extension": 52,
        "partial": 11,
        "unmapped": 5
      },
      "core_explicit_mapping_rate": 0.4728682170542636,
      "extended_explicit_mapping_rate": 0.875968992248062,
      "partial_or_state_not_counted": 11,
      "unmapped": 5
    },
    "tie_the_bouquet_of_fake_flowers": {
      "n": 623,
      "counts": {
        "mapped_core": 407,
        "mapped_extension": 143,
        "partial": 72,
        "unmapped": 1
      },
      "core_explicit_mapping_rate": 0.6532905296950241,
      "extended_explicit_mapping_rate": 0.8828250401284109,
      "partial_or_state_not_counted": 72,
      "unmapped": 1
    },
    "tie_the_shoes": {
      "n": 348,
      "counts": {
        "partial": 76,
        "mapped_extension": 72,
        "mapped_core": 200
      },
      "core_explicit_mapping_rate": 0.5747126436781609,
      "extended_explicit_mapping_rate": 0.7816091954022989,
      "partial_or_state_not_counted": 76,
      "unmapped": 0
    },
    "untangle_the_cables": {
      "n": 378,
      "counts": {
        "partial": 102,
        "mapped_core": 251,
        "mapped_extension": 21,
        "unmapped": 4
      },
      "core_explicit_mapping_rate": 0.6640211640211641,
      "extended_explicit_mapping_rate": 0.7195767195767195,
      "partial_or_state_not_counted": 102,
      "unmapped": 4
    },
    "untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food": {
      "n": 25,
      "counts": {
        "mapped_core": 21,
        "mapped_extension": 2,
        "partial": 2
      },
      "core_explicit_mapping_rate": 0.84,
      "extended_explicit_mapping_rate": 0.92,
      "partial_or_state_not_counted": 2,
      "unmapped": 0
    },
    "untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit": {
      "n": 183,
      "counts": {
        "mapped_core": 114,
        "partial": 28,
        "mapped_extension": 15,
        "unmapped": 26
      },
      "core_explicit_mapping_rate": 0.6229508196721312,
      "extended_explicit_mapping_rate": 0.7049180327868853,
      "partial_or_state_not_counted": 28,
      "unmapped": 26
    },
    "unzip_the_laundry_bag_and_remove_the_bra": {
      "n": 205,
      "counts": {
        "mapped_core": 93,
        "mapped_extension": 85,
        "partial": 27
      },
      "core_explicit_mapping_rate": 0.45365853658536587,
      "extended_explicit_mapping_rate": 0.8682926829268293,
      "partial_or_state_not_counted": 27,
      "unmapped": 0
    },
    "wrap_the_cables": {
      "n": 30,
      "counts": {
        "mapped_core": 18,
        "unmapped": 6,
        "mapped_extension": 4,
        "partial": 2
      },
      "core_explicit_mapping_rate": 0.6,
      "extended_explicit_mapping_rate": 0.7333333333333333,
      "partial_or_state_not_counted": 2,
      "unmapped": 6
    },
    "wrap_the_headphones": {
      "n": 46,
      "counts": {
        "mapped_core": 28,
        "partial": 1,
        "mapped_extension": 14,
        "unmapped": 3
      },
      "core_explicit_mapping_rate": 0.6086956521739131,
      "extended_explicit_mapping_rate": 0.9130434782608695,
      "partial_or_state_not_counted": 1,
      "unmapped": 3
    },
    "zip_tie_the_wires": {
      "n": 274,
      "counts": {
        "mapped_extension": 79,
        "mapped_core": 110,
        "partial": 84,
        "unmapped": 1
      },
      "core_explicit_mapping_rate": 0.40145985401459855,
      "extended_explicit_mapping_rate": 0.6897810218978102,
      "partial_or_state_not_counted": 84,
      "unmapped": 1
    },
    "zip_up_the_jacket": {
      "n": 46,
      "counts": {
        "mapped_extension": 9,
        "mapped_core": 21,
        "partial": 6,
        "unmapped": 10
      },
      "core_explicit_mapping_rate": 0.45652173913043476,
      "extended_explicit_mapping_rate": 0.6521739130434783,
      "partial_or_state_not_counted": 6,
      "unmapped": 10
    }
  },
  "yam_tasks": {
    "tube_insertion_recovery": {
      "n": 80,
      "counts": {
        "mapped_extension": 20,
        "state_only": 60
      },
      "core_explicit_mapping_rate": 0.0,
      "extended_explicit_mapping_rate": 0.25,
      "partial_or_state_not_counted": 60,
      "unmapped": 0
    },
    "tube_pick_and_insert": {
      "n": 116,
      "counts": {
        "mapped_core": 58,
        "state_only": 58
      },
      "core_explicit_mapping_rate": 0.5,
      "extended_explicit_mapping_rate": 0.5,
      "partial_or_state_not_counted": 58,
      "unmapped": 0
    }
  },
  "scope": "descriptive coverage of this input, not accuracy/generalization success; rule design used the same corpus, no held-out test",
  "definition": "mapped_core/extension = explicit text action intent representable; partial/state_only/unmapped excluded from both numerator; all segments retained in denominator"
}

ABC原始标签中的adjust作为partial候选，tie/untie/install作为未展开目标宏；不把任何动词都塞进recover。新增hold、transport、rotate、fold_unfold、straighten、wind、cross、press、open_close、zip_slide分别表达持续支撑、运输、旋转、折叠、理直、绕卷、交叉、按压、开合和拉链滑动。knot、unknot_or_disentangle、connect_joint仍需任务拓扑/接触机制模板，未当成已解决的原子动作。

yam全部49 clip有显式动作提及，但完整任务动作链可由叙述确定的为0/49：29个常规clip仅明确reach→retreat，中间抬起/近邻为候选；20个recovery类clip只明确rotate，其余是静态姿态，不能证明失败与恢复。196条叙述只有16种原句，重复文风使按条覆盖偏乐观，不构成49种独立泛化证据。

典型未映射：adjust（无目标）、trim（无对象/工具）、Reach into the gripper in the takeout bag（角色错置）、Pick up the center of the tabletop from the right side of the ribbon（错误对象）。部分映射：Tie the shoelace on the shoe未区分结型/穿绕顺序；Install the adapter on the hose未说明旋拧或压配；The right gripper is close to the blue-capped tube at the rack不能区分align/insert。完整逐条缺口见coverage_gaps.jsonl。

## 4. 每任务实例化示例
以下均为设计计划，不冒充数据中的实际执行顺序；含未展开宏的任务尚需补充分支/结型。16个ABC任务、2个yam任务类别全覆盖，另列B/C设计例但不计实测覆盖。

### connect_and_route_the_hose
原语：grasp_lift → align → rotate → transport → place_release。
问题：夹爪是否夹住软管接头？；接头是否对准软管口？；连接是否通过指定防漏/拉脱检验？
槽位：{"object": "软管", "part": "接头", "opening": "软管口", "mechanism": "连接方式待确认；若非螺纹不使用rotate"}
范围：in_scope
### place_the_bra_in_the_laundry_bag_and_zip_it_closed
原语：grasp_lift → align → zip_slide → place_release。
问题：拉头是否沿洗衣袋拉链到达指定端点？；链齿是否按任务要求打开或闭合？；衣物是否已在指定袋内或接收区且脱爪？
槽位：{"object": "洗衣袋拉链", "closure_state": "unzip任务为打开，其余为闭合"}
范围：in_scope
### place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot
原语：open_close → grasp_lift → place_release → cross → wind → pull → place_release。
问题：食品是否已位于指定袋内？；袋口两部分是否形成要求的结型关系？；卸载后袋口是否保持要求的闭合状态？
槽位：{"object": "食品与袋口", "gap": "复杂结或普通结由任务协议指定，示例不保证可执行"}
范围：in_scope
### place_the_food_into_the_grocery_bag_and_tie_it
原语：open_close → grasp_lift → place_release → cross → wind → pull → place_release。
问题：食品是否已位于指定袋内？；袋口两部分是否形成要求的结型关系？；卸载后袋口是否保持要求的闭合状态？
槽位：{"object": "食品与袋口", "gap": "复杂结或普通结由任务协议指定，示例不保证可执行"}
范围：in_scope
### place_the_mixed_food_into_the_grocery_bags_and_tie_them
原语：open_close → grasp_lift → place_release → cross → wind → pull → place_release。
问题：食品是否已位于指定袋内？；袋口两部分是否形成要求的结型关系？；卸载后袋口是否保持要求的闭合状态？
槽位：{"object": "食品与袋口", "gap": "复杂结或普通结由任务协议指定，示例不保证可执行"}
范围：in_scope
### roll_the_ties
原语：grasp_lift → straighten → wind → place_release。
问题：领带是否已展平？；领带是否形成指定卷形？；领带是否留在收纳盒且与夹爪分离？
槽位：{"object": "宽领带", "shape": "卷形", "destination": "收纳盒"}
范围：in_scope
### tie_the_bouquet_of_fake_flowers
原语：grasp_lift → place_release → fold_unfold → press → cross → wind → pull → place_release。
问题：包装纸是否沿指定折线包住花束？；丝带两端是否形成指定上下交叉？；打结方案完成后环结是否在卸载时保留？
槽位：{"object": "花束及包装纸", "part": "丝带两端", "gap": "结型及穿绕子计划未由tie标签给出"}
范围：in_scope
### tie_the_shoes
原语：grasp_lift → cross → wind → pull → place_release。
问题：鞋带两端是否形成指定交叉？；自由端是否穿过指定鞋带环？；拉动后要求的双环结是否保留？
槽位：{"object": "鞋带", "opening": "指定鞋带环", "gap": "结型/穿绕顺序需补充，wind不等同打结完成"}
范围：in_scope
### untangle_the_cables
原语：grasp_lift → hold → pull → straighten → place_release。
问题：选中线缆是否可与其余线缆区分？；该线缆是否已脱离指定缠绕关系？；线缆是否在容器中且已脱爪？
槽位：{"object": "选中线缆", "gap": "解缠拓扑与动作选择依当前构型"}
范围：in_scope
### untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food
原语：grasp_lift → pull → open_close → grasp_lift → place_release。
问题：袋口的结是否已解除到可展开状态？；袋口是否张开？；取出的食品是否落在接收区且脱爪？
槽位：{"object": "袋结与食品", "opening": "袋口", "gap": "解结子计划待当前拓扑解析"}
范围：in_scope
### untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit
原语：grasp_lift → pull → open_close → grasp_lift → place_release。
问题：袋口的结是否已解除到可展开状态？；袋口是否张开？；取出的食品是否落在接收区且脱爪？
槽位：{"object": "袋结与食品", "opening": "袋口", "gap": "解结子计划待当前拓扑解析"}
范围：in_scope
### unzip_the_laundry_bag_and_remove_the_bra
原语：grasp_lift → align → zip_slide → place_release。
问题：拉头是否沿洗衣袋拉链到达指定端点？；链齿是否按任务要求打开或闭合？；衣物是否已在指定袋内或接收区且脱爪？
槽位：{"object": "洗衣袋拉链", "closure_state": "unzip任务为打开，其余为闭合"}
范围：in_scope
### wrap_the_cables
原语：grasp_lift → hold → wind → place_release。
问题：线缆是否连续绕过指定卷轴/耳机主体？；绕卷是否在松手后保留？
槽位：{"object": "线缆", "reference": "线盘或耳机主体", "shape": "指定绕卷"}
范围：in_scope
### wrap_the_headphones
原语：grasp_lift → hold → wind → place_release。
问题：线缆是否连续绕过指定卷轴/耳机主体？；绕卷是否在松手后保留？
槽位：{"object": "线缆", "reference": "线盘或耳机主体", "shape": "指定绕卷"}
范围：in_scope
### zip_tie_the_wires
原语：grasp_lift → wind → align → insert → regrasp → pull → place_release。
问题：扎带是否围住本轮线束？；同根尾端是否从锁头出口穿出？；牵拉后扎带与线束包围关系是否保留？
槽位：{"object": "扎带", "reference": "线束", "opening": "锁头入口", "note": "此为任务设计例，原段未必细述insert，不从task名补入覆盖"}
范围：in_scope
### zip_up_the_jacket
原语：grasp_lift → align → zip_slide → place_release。
问题：拉头是否沿夹克拉链到达指定端点？；链齿是否按任务要求打开或闭合？；衣物是否已在指定袋内或接收区且脱爪？
槽位：{"object": "夹克拉链", "closure_state": "unzip任务为打开，其余为闭合"}
范围：in_scope
### tube_pick_and_insert
原语：reach → grasp_lift → transport → align → insert → place_release → retreat。
问题：指定手是否夹住本轮蓝盖试管？；试管轴是否对准橙色管架的指定孔口？；管底是否跨过该孔入口？；夹爪撤离后试管是否由管架支撑？
槽位：{"arm": "按clip observed_active_arm绑定", "object": "蓝盖试管", "opening": "未编号，需任务指定", "depth": "当前文本不能确认"}
范围：in_scope
### tube_insertion_recovery
原语：check → recover → rotate → align → insert → place_release → retreat。
问题：是否有可见证据表明上次插入失败？；试管是否退到孔口外且有间隙？；调整后试管轴是否对齐孔口？
槽位：{"object": "同一试管", "attempt_id": "重试递增；不得仅凭task_type判失败"}
范围：in_scope
### E89_B_insert_plug
原语：reach → grasp_lift → align → insert → place_release。
问题：插头是否对准指定插座？；插头是否达到规定插入位置？；电接触是否有独立电气信号证明？
槽位：{"object": "插头", "opening": "插座", "hidden": "导通"}
范围：design_only_no_E91_B_segments
### E89_C_fold_towel
原语：grasp_lift → fold_unfold → press → place_release。
问题：毛巾两边是否按指定折线重合？；折叠毛巾是否放在指定叠放层且脱爪？
槽位：{"object": "毛巾", "shape": "任务指定折叠层数"}
范围：design_only_no_E91_C_segments

## 5. 能迁移什么，不能由本轮证明什么
原语定义、四值答案域、带类型槽位、同物体身份/因果历史、unknown不通行、失败重试与新一轮分开，可作为新任务零样本设计骨架迁移。任务大脑可由指令产生候选计划，但必须保留并行支撑、循环、可选分支和不可见状态，不能照搬Piper十阶段线性顺序。连接软管可能需要旋拧，扎带围线束与自穿成环也有不同对象关系。

VLM能否零样本回答具体问题需要新任务真实图像检验。本轮只证明部分文本可表达性，没有运行VLM视觉推理、分类头或机器人，不能声称零样本视觉准确率。明确对象/关系的提示可能复用，细孔、线缆身份、深度、遮挡、视角变化与新夹爪仍需任务验证；“朝下”不是“对准”，“在孔旁”不是“已插入”。状态模板可表达静态布局，但不能因此追认此前发生的动作。

固定Piper 10类小分类头不能直接解释新任务的fold/zip_slide等新类；需要更换/扩展标签空间及输出头，或使用条件化通用谓词头并在新任务训练parent上验证。共用held/near等模板的头可初始化迁移，仍需新域正/负/unknown、失败/恢复样本、相机与对象尺度校准；不默认所有任务都必须从头训练，也不默认共享头无需适配。重标对象绑定、任务图/分支、可见完成条件、局部阶段边界与失败类型；不能将粗tie标签硬切为精确动作段。模型训练标签可作辅助，但不能自动升级为测试真值。

不可见状态需要任务特定信号：抓持需开度/电流结合物体随动；锁紧需校准载荷—位移或力矩证据；插头导通需电气测量；软管防漏需压力/流量检测；试管插入深度需可标定几何、位姿或接触证据。q或开爪命令本身不证明持物/释放，二维图像重叠不证明三维进入。没有信号就unknown，不强造新视觉标签。世界头仍不读取模型评估语言。

## 6. 新任务上线最小流程
1. 冻结任务指令、物体/部件词典、真实数据parent划分与源时间轴。列出视觉完成和物理功能完成的不同验收条件，排除demo3 episode142；不引入仿真/retarget或平移。
2. 用训练parent文本/少量真实片段实例化原语图与谓词槽位，显式写AND/OR守卫、支撑不变量、目标身份、重试/终止条件。优先复用已有模板；结型、螺纹连接等若无法表达，先补机制，不把笼统动作词算覆盖。
3. 在训练parent补标各分支、失败和unknown，保留因果截止、实际可见性及边界区间。先检验零样本VLM可回答性，再决定小头的共享/扩展/微调；样本量按稀有分支覆盖决定，本轮不凭模型文本指定阈值。
4. 锁定评估协议后进行真实图像与必要本体信号评估，按parent聚合逐谓词混淆、unknown率、失败漏报、阶段边界和任务终止误报，另分可见性/新对象/新视角。不用本轮规则覆盖率选checkpoint。只有E98 B1选定20 dev parent的阶段/谓词可按授权用模型真值，并须在结论写“基于模型标注的dev真值”；此例外不能扩展到新任务dev/test。
5. 谓词与守卫通过独立验收后才小规模实机上线；保留unknown等待/补观察及恢复分支。文本体系通过不等于策略或硬件控制器已验证。若要机械成功，先由用户指定检验协议和可用信号。

## 7. 本轮结论与未完成项
跨任务更适合复用“原语+关系谓词”，不能复用Piper固定阶段编号作为所有任务的分类标签。10项扩展覆盖可明确表达的新动作；打结/解缠/接头安装仍需拓扑或机制子计划。当前ABC仅A类且动作叙述重复，yam仅试管两类且16种原句，因此跨任务视觉/控制泛化尚未被实验验证。未执行训练、GPU、视频解码、测试标注或机器人动作，这些不是本轮文本交付的缺漏。
需要后续用户决定：新任务的物理验收条件（例如锁紧载荷、插入深度、接头防漏）以及是否另开真实图像/本体泛化实验；本轮设计与全量映射无需等待该决定。


## 8. 执行与核验
CPU Python标准库，GPU 0h，无模型加载/下载，无大文件复制，无训练。代码与产物仅写data/e98_generalization_v1及本报告，另对共享日志做带锁追加。源文件前后SHA256一致，21谓词/10阶段引用、全部槽位绑定、2,484/196逐条记录与任务汇总一致；核验见validation.json。脚本为本目录build.py，命令python3 data/e98_generalization_v1/build.py；输出状态与交付哈希见status.json和delivery_manifest.json。

## 9. 最终审阅摘要与逐任务覆盖

|语料|分母|核心显式映射|扩展后显式映射|其余|
|---|---:|---:|---:|---|
|ABC cleaned段|2,484|1,424（57.33%）|1,962（78.99%）|459部分、63未映射|
|yam叙述|196|58（29.59%）|78（39.80%）|118静态状态|

ABC去重原文349种：核心178（51.00%），扩展后274（78.51%）。yam去重原文16种：核心4、扩展后6（37.50%）；按条比率受重复文案影响。两个分母不混合。ABC总记录116、非空98，16任务中15有段；空任务不计算率。

|任务|段数|核心覆盖|扩展后覆盖|部分/未映射|
|---|---:|---:|---:|---:|
|connect_and_route_the_hose|51|52.94%|66.67%|10/7|
|place_the_bra_in_the_laundry_bag_and_zip_it_closed|72|30.56%|73.61%|19/0|
|place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot|34|73.53%|82.35%|6/0|
|place_the_food_into_the_grocery_bag_and_tie_it|40|65.00%|67.50%|13/0|
|place_the_mixed_food_into_the_grocery_bags_and_tie_them|0|缺数据|缺数据|0/0|
|roll_the_ties|129|47.29%|87.60%|11/5|
|tie_the_bouquet_of_fake_flowers|623|65.33%|88.28%|72/1|
|tie_the_shoes|348|57.47%|78.16%|76/0|
|untangle_the_cables|378|66.40%|71.96%|102/4|
|untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food|25|84.00%|92.00%|2/0|
|untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit|183|62.30%|70.49%|28/26|
|unzip_the_laundry_bag_and_remove_the_bra|205|45.37%|86.83%|27/0|
|wrap_the_cables|30|60.00%|73.33%|2/6|
|wrap_the_headphones|46|60.87%|91.30%|1/3|
|zip_tie_the_wires|274|40.15%|68.98%|84/1|
|zip_up_the_jacket|46|45.65%|65.22%|6/10|

部分项规则计数：{"goal_macro": 228, "adjust_unspecified": 230, "move_ambiguous": 1}；目标宏计数：{"connect_joint": 10, "knot": 93, "unknot_or_disentangle": 125}。

缺口模板提案见semantic_contract_supplement.json：结型交叉图、解除缠绕、接头机械啮合/功能检验。尚未定义任务具体拓扑/接触机制，故不把这些提案计入78.99%分子。Grab仅选grasp_only变体，不声称lift发生；Place是放置意图，不声称已观测释放；预条件模板是计划条件，不是文本自动推断的真值。

报告日期按服务器实际时钟换算BRT为2026-11-14。用户给定E98规则变更标注日期2026-11-15仍按明确授权执行，未自行更改来源记录；本会话没有使用dev例外。

最终复现入口为build_v2.py；build.py及draft_01_count_assertion保留首次计数断言失败过程，不应作为最终覆盖结果。两次均为CPU秒级程序，无长任务worker、GPU或失败GPUh；初始只读路径定位已结束。

共享EXPERIMENT_LOG与worklog已于2026-11-14T14:11:18.524560-03:00 BRT带锁追加E98-G条目。最终核验final_validation.json通过；资源新增约10.00 MB（最终manifest自身及日志追加为小量额外），远低于60GB。
