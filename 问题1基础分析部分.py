# 导入会使用到的库
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# 设置Seaborn图表样式和字体
sns.set(font="simhei", style="whitegrid", font_scale=1.6)

# 读取数据
df_male = pd.read_excel('附件1：1名男大学生的一日食谱.xlsx').fillna(method='ffill')  # 读取男大学生食谱数据，并填充空值
df_female = pd.read_excel('附件2：1名女大学生的一日食谱.xlsx').fillna(method='ffill')  # 读取女大学生食谱数据，并填充空值
df_canteen = pd.read_excel('附件3：某高校学生食堂一日三餐主要食物信息统计表.xlsx')  # 读取食堂食物信息统计表

# 获取成分含量文件列表的前四个文件
component_files = os.listdir('成分含量')[:4]
df_components = pd.DataFrame(columns=['食品'])  # 创建空的DataFrame，包含“食品”列

# 合并所有成分含量文件的数据
for file in component_files:
    df_components = pd.merge(df_components, pd.read_excel(f'成分含量\\{file}'), on='食品', how='right')

# 定义要保留的列名
columns_to_keep = '''食品、蛋白质、脂肪、碳水化合物、膳食纤维、酒精（乙醇）、钙、铁、锌、维生素A、维生素B1、维生素B2、维生素C、异亮氨酸、亮氨酸、赖氨酸、含硫氨基酸、芳香族氨基酸、苏氨酸、色氨酸、缬氨酸'''.split('、')

# 过滤并保留上述列
df_components = df_components[columns_to_keep]

# 将氨基酸含量数据按百分比转换
for amino_acid in ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']:
    df_components[amino_acid] = df_components[amino_acid] * 100

# 打印df_components的内容
print(df_components)

# 绘制每种成分的水平条形图
for nutrient in df_components.columns[1:]:
    # 生成数据
    categories = df_components['食品']  # 食品名称
    values = df_components[nutrient]  # 每种食品的特定营养成分含量

    # 设置图表大小
    plt.figure(figsize=(8, 12))
    
    # 创建水平条形图
    plt.barh(categories, values, color='lightblue', edgecolor='black')
    
    # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(range(len(categories)), categories, fontsize=8)
    
    # 添加标题和标签
    plt.xlabel(f'每100克{nutrient}含量', fontsize=14)
    plt.ylabel('食品', fontsize=14)
    plt.title(f'{nutrient}含量水平条形图', fontsize=16)
    
    # 保存图形到文件
    plt.savefig(f'问题1\\各食物的营养成分比例（用于饮食计划的调整\\{nutrient}.png', dpi=500, bbox_inches='tight')
    
    # 显示图形
    plt.show()

# 计算每种食品在食谱中的营养成分
nutrient_list = []
cols = df_components.columns[1:]
for food, weight, portions in zip(df_male['主要成分'], df_male['可食部（克/份）'], df_male['食用份数']):
    nutrient_list.append((df_components[df_components.食品 == food][cols].values * weight / 100 * portions)[0])
temp_df = pd.DataFrame(nutrient_list, columns=cols)
result_df = pd.concat([df_male, temp_df], axis=1)
print(result_df)

# 定义宏量营养素及其对应能量系数
macro_nutrients = ['蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）']
energy_factors = [4, 9, 4, 2, 7]

# 计算各宏量营养素提供的能量
temp_energy = result_df[macro_nutrients] * np.array(energy_factors)
temp_energy.columns = [f'{i}能量' for i in temp_energy.columns]

# 将能量数据合并到结果数据中
result_df = pd.concat([result_df, temp_energy], axis=1)
print(result_df)

# 定义食品类别字典
food_category_mapping = {
    '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品', '黄豆': '奶、干豆、坚果、种子类及制品',
    '稻米': '谷、薯类', '小米': '谷、薯类', '小麦粉': '谷、薯类', '豆油': '植物油类', '鸡蛋': '畜、禽、鱼、蛋类及制品',
    '地瓜': '谷、薯类', '南瓜': '蔬菜、菌藻、水果类', '猪肉瘦': '畜、禽、鱼、蛋类及制品', '鸡肉': '畜、禽、鱼、蛋类及制品',
    '油菜': '蔬菜、菌藻、水果类', '猪肉': '畜、禽、鱼、蛋类及制品', '白菜': '蔬菜、菌藻、水果类', '牛肉': '畜、禽、鱼、蛋类及制品',
    '胡萝卜': '蔬菜、菌藻、水果类', '火腿肠': '畜、禽、鱼、蛋类及制品', '土豆': '谷、薯类', '芹菜': '蔬菜、菌藻、水果类',
    '韭菜': '蔬菜、菌藻、水果类', '菠菜': '蔬菜、菌藻、水果类', '芝麻油': '植物油类', '海带': '蔬菜、菌藻、水果类',
    '豆腐': '奶、干豆、坚果、种子类及制品', '干豆腐': '奶、干豆、坚果、种子类及制品', '木耳': '蔬菜、菌藻、水果类',
    '花生米': '奶、干豆、坚果、种子类及制品', '苹果': '蔬菜、菌藻、水果类', '橙': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类',
    '荞麦面': '谷、薯类', '洋葱': '蔬菜、菌藻、水果类', '西红柿': '蔬菜、菌藻、水果类', '紫菜': '蔬菜、菌藻、水果类',
    '萝卜': '蔬菜、菌藻、水果类', '粉条': '谷、薯类', '鱼丸': '畜、禽、鱼、蛋类及制品', '明太鱼': '畜、禽、鱼、蛋类及制品',
    '香菇': '蔬菜、菌藻、水果类', '卷心菜': '蔬菜、菌藻、水果类', '青椒': '蔬菜、菌藻、水果类', '豆芽': '奶、干豆、坚果、种子类及制品',
    '黄瓜': '蔬菜、菌藻、水果类', '茄子': '蔬菜、菌藻、水果类', '扁豆': '奶、干豆、坚果、种子类及制品', '蒜台': '蔬菜、菌藻、水果类',
    '杏鲍菇': '蔬菜、菌藻、水果类', '酸菜': '蔬菜、菌藻、水果类', '五花猪肉': '畜、禽、鱼、蛋类及制品', '猪排骨': '畜、禽、鱼、蛋类及制品',
    '炸鸡块': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '黄花鱼': '畜、禽、鱼、蛋类及制品',
    '带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类',
    '玉米面': '谷、薯类', '柚子': '蔬菜、菌藻、水果类'
}

# 计算食品类别数量
def count_food_categories(food_list):
    total_count = len(food_list)
    category_counts = {
        '谷、薯类': 0,
        '蔬菜、菌藻、水果类': 0,
        '畜、禽、鱼、蛋类及制品': 0,
        '奶、干豆、坚果、种子类及制品': 0,
        '植物油类': 0
    }

    for food in food_list:
        category = food_category_mapping.get(food)
        if category:
            category_counts[category] += 1

    return total_count, category_counts

# 获取独特的主要成分列表
unique_food_list = result_df['主要成分'].unique()

# 计算并输出食品类别数量
total_food_count, category_count_dict = count_food_categories(unique_food_list)
print(f"总的食物数量: {total_food_count}")
print("不同类别食物的数量:")
for category, count in category_count_dict.items():
    print(f"{category}: {count}")

# 绘制食品类别数量的饼图
category_labels = category_count_dict.keys()
category_sizes = category_count_dict.values()
category_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

# 设置图表大小
plt.figure(figsize=(10, 10))
# 创建饼图
plt.pie(category_sizes, labels=category_labels, colors=category_colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('不同类别食物的数量分布', fontsize=16)
plt.savefig(f'问题1\\同学1谷物摄入情况.png', dpi=500, bbox_inches='tight')
# 显示图形
plt.show()

print(result_df.columns)

# 宏量营养素供能占比
# 大学生群体每日宏量营养素供能占总能量的百分比参考值分别为：
# 蛋白质10%-15%、脂肪20%-30%、碳水化合物50%-65%
energy_columns = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']

# 计算各能量成分的占比
energy_distribution = (result_df[energy_columns].sum() / result_df[energy_columns].sum().sum())
energy_distribution.to_excel('问题1\\同学1的日能量贡献.xlsx')

# 定义数据
energy_categories = energy_distribution.index
energy_values = energy_distribution.values
# 设置图表大小
plt.figure(figsize=(12, 8))
# 创建条形图
plt.bar(energy_categories, energy_values, color='lightblue', edgecolor='black')
# 添加标题和标签
plt.xlabel('摄入能量成分', fontsize=14)
plt.ylabel('能量占比', fontsize=14)
plt.title('同学1每日能量摄入成分占比', fontsize=16)
# 设置刻度标签大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题1\\同学1能量摄入情况.png', dpi=500, bbox_inches='tight')
# 显示图形
plt.show()
print('日总能量摄入量', result_df[energy_columns].sum().sum())

print(energy_distribution)

# 计算非产能主要营养素摄入量
micronutrients = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
micronutrient_totals = (result_df[micronutrients].sum())
micronutrient_totals.to_excel('问题1\\同学1每日膳食非产能主要营养素参考摄入量.xlsx')

# 绘制非产能主要营养素摄入量条形图
micronutrient_categories = micronutrient_totals.index
micronutrient_values = micronutrient_totals.values
# 设置图表大小
plt.figure(figsize=(12, 8))
# 创建条形图
plt.bar(micronutrient_categories, micronutrient_values, color='lightblue', edgecolor='black')
# 添加标题和标签
plt.xlabel('非产能营养素成分', fontsize=14)
plt.ylabel('摄入量', fontsize=14)
plt.title('同学1每日非产能主要营养素摄入量', fontsize=16)
# 设置刻度标签大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题1\\同学1非产能摄入情况.png', dpi=500, bbox_inches='tight')
# 显示图形
plt.show()

print(micronutrient_totals)

# 计算每种餐类的氨基酸评分
for meal_type in result_df['餐类'].unique():
    meal_df = result_df[result_df.餐类 == meal_type]
    
    # 计算平均混合食物蛋白质的AAS评分
    amino_acids = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    reference_values = [40, 70, 55, 35, 60, 40, 10, 50]
    
    # 计算并保存氨基酸摄入量
    (result_df[amino_acids].sum() / result_df['蛋白质'].sum() * 10).to_excel('问题1\\同学1氨酸摄入情况.xlsx')
    
    # 计算并打印AAS评分
    AAS_score = ((meal_df[amino_acids].sum() / meal_df['蛋白质'].sum() * 10) / np.array(reference_values) * 100).sum() / 8
    print(f'该同学{meal_type}的AAS摄入评分为', AAS_score)

# 打印氨基酸摄入量
print(result_df[amino_acids].sum() / result_df['蛋白质'].sum() * 10)
