import pandas as pd
import os

def generate_debug_csv():
    # 路径配置
    input_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
    output_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/debug_check.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件: {input_path}")
        return

    # 读取数据
    df = pd.read_csv(input_path)
    
    # 提取关键列进行核对
    # index: 文件名
    # cut_points: 原本o6里的真值
    # cut_point: 从multi.csv解析出来的对比值
    # ltx_cutpoint_numloss: 计算出的段数差
    columns = ['index', 'cut_points', 'cut_point', 'ltx_cutpoint_numloss', 'cut_point_numloss_77']
    
    # 过滤出存在对比数据的行（剔除掉空值行，方便核对）
    debug_df = df[df['cut_point'].notna()][columns]
    
    # 保存
    debug_df.to_csv(output_path, index=False)
    
    print(f"✅ 调试文件已生成: {output_path}")
    print(f"📊 共提取了 {len(debug_df)} 条有效对比数据，请开始人工核对。")

if __name__ == "__main__":
    generate_debug_csv()