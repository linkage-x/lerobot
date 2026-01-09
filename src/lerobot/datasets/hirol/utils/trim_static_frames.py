# 功能是将类似/data/2025/unitree_co-train/bread_picking/1227_duo_unitree_bread_picking_human_182ep/episode_0002/data.json中的静止帧去掉.
# 根据data[k].ee_states.right.pose的变化情况来判断静止帧.
# 其中，需要去掉的是轨迹开始时末端执行器在连续的若干帧中位置变化较小的帧. 基于窗口扫描方法判断是否有明显移动，如果把明显移动的帧标记为"+"，静止的帧标记为"-"，窗口左右边界记为[]，则类似下面这样：
# --------------------[----]++++++++++++++++++++++++++
# 去除[左侧的帧，保留右侧的帧.
# 将处理后的数据保存为新的json文件，例如/data/2025/unitree_co-train/bread_picking/1227_duo_unitree_bread_picking_human_182ep_trimmed/episode_0002/data.json,并拷贝相应的colors和depth文件夹.
# 注意，这个预处理脚本还需过滤left，head相关字段&&图片，仅保留right相关字段&&图片.
# 例如输入/data/2025/unitree_co-train/bread_picking/1227_duo_unitree_bread_picking_human_182ep。输出处理后的数据到/data/2025/unitree_co-train/bread_picking/1227_duo_unitree_bread_picking_human_182ep_trimmed