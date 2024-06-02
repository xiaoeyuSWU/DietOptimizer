import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import dual_annealing

# 忽略提醒
warnings.filterwarnings("ignore")

# 设置绘图参数
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴的负号显示问题

# 读取附件3的数据
df1 = pd.read_excel('附件3：某高校学生食堂一日三餐主要食物信息统计表.xlsx').fillna(method='ffill')

# 获取成分含量文件列表
a = os.listdir('成分含量')[:4]
df2 = pd.DataFrame(columns=['食品'])

# 合并成分含量数据
for i in a:
    df2 = pd.merge(df2, pd.read_excel(f'成分含量\\{i}'), on='食品', how='right')

# 指定需要的列
columns_needed = '''食品、蛋白质、脂肪、碳水化合物、膳食纤维、酒精（乙醇）、钙、铁、锌、维生素A、维生素B1、维生素B2、维生素C、异亮氨酸、亮氨酸、赖氨酸、含硫氨基酸、芳香族氨基酸、苏氨酸、色氨酸、缬氨酸'''
columns_needed = columns_needed.split('、')
df2 = df2[columns_needed]

# 将特定列的值乘以100
for col in ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']:
    df2[col] = df2[col] * 100

# 计算各食物的营养成分及对应的能量
ls = []
col = df2.columns[1:]
for pca, weight in zip(df1['主要成分'], df1['可食部（克/份）']):
    ls.append((df2[df2.食品 == pca][col].values * weight / 100)[0])

tem = pd.DataFrame(ls, columns=col)
result = pd.concat([df1, tem], axis=1)

# 计算能量
col = ['蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）']
energy_factors = [4, 9, 4, 2, 7]

tem = result[col] * np.array(energy_factors)
eng_col = [i + '能量' for i in tem.columns]
tem.columns = eng_col
result = pd.concat([result, tem], axis=1)
result['总能量'] = result[eng_col].sum(axis=1).values

# 按照餐类、序号和食物名称分组计算总和
t = result.groupby(['餐类', '序号', '食物名称'])[['可食部（克/份）', '价格（元/份）', '是否\n可半份', '蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C', '异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸', '蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量', '总能量']].sum()

# 为每一天添加星期列并合并结果
total = pd.DataFrame()
for i in range(1, 8):
    t['星期'] = i
    total = pd.concat([total, t])

# 保存结果到Excel
output_path = '问题3\\优化模型数据文件.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
total.to_excel(output_path)

print("数据处理完成，并已保存到", output_path)

# 读取数据
df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')

# 定义食物类别字典
food_categories = {
    '大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类',
    '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品',
    '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类',
    '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品',
    '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品',
    '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品',
    '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品',
    '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类',
    '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类',
    '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品',
    '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品',
    '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品',
    '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品',
    '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类',
    '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类', '牛奶': '奶、干豆、坚果、种子类及制品',
    '酸奶': '奶、干豆、坚果、种子类及制品', '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品',
    '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类', '拌海带丝': '蔬菜、菌藻、水果类',
    '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类', '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'
}

# 添加类别信息
df['类别'] = [food_categories[i] for i in df['食物名称']]

# 定义退火算法目标函数
def objective(x):
    total_cost = np.sum(prices * x)
    total_calories = np.sum(calories * x)
    total_protein = np.sum(protein * x)
    total_fat = np.sum(fat * x)
    total_carbs = np.sum(carbs * x)
    total_calcium = np.sum(calcium * x)
    total_iron = np.sum(iron * x)
    total_zinc = np.sum(zinc * x)
    total_vitamin_a = np.sum(vitamin_a * x)
    total_vitamin_b1 = np.sum(vitamin_b1 * x)
    total_vitamin_b2 = np.sum(vitamin_b2 * x)
    total_vitamin_c = np.sum(vitamin_c * x)

    # 计算宏量营养素能量占比
    total_energy = 4 * total_protein + 9 * total_fat + 4 * total_carbs
    protein_energy_ratio = (4 * total_protein) / total_energy
    fat_energy_ratio = (9 * total_fat) / total_energy
    carbs_energy_ratio = (4 * total_carbs) / total_energy

    # 计算约束违反情况
    constraints_violation = 0
    constraints_violation += 0 if 1800 <= total_calories <= 2000 else 1
    constraints_violation += 0 if 0.10 <= protein_energy_ratio <= 0.15 else 1
    constraints_violation += 0 if 0.20 <= fat_energy_ratio <= 0.30 else 1
    constraints_violation += 0 if 0.50 <= carbs_energy_ratio <= 0.65 else 1
    constraints_violation += 0 if 700 <= total_calcium <= 900 else 1
    constraints_violation += 0 if 16 <= total_iron <= 24 else 1
    constraints_violation += 0 if 5.5 <= total_zinc <= 10 else 1
    constraints_violation += 0 if 600 <= total_vitamin_a <= 800 else 1
    constraints_violation += 0 if 0.9 <= total_vitamin_b1 <= 1.5 else 1
    constraints_violation += 0 if 0.9 <= total_vitamin_b2 <= 1.5 else 1
    constraints_violation += 0 if 80 <= total_vitamin_c <= 120 else 1

    return total_cost + constraints_violation * 1000  # 对违反约束的情况施加较大的惩罚

# 提取相关数据列用于优化
prices = df['价格（元/份）'].values
calories = df['总能量'].values
protein = df['蛋白质'].values
fat = df['脂肪'].values
carbs = df['碳水化合物'].values
calcium = df['钙'].values
iron = df['铁'].values
zinc = df['锌'].values
vitamin_a = df['维生素A'].values
vitamin_b1 = df['维生素B1'].values
vitamin_b2 = df['维生素B2'].values
vitamin_c = df['维生素C'].values

# 定义变量的上下界
bounds = [(0, 2) for _ in range(len(prices))]

# 退火算法求解
result = dual_annealing(objective, bounds, maxiter=1000)

# 获取最优解
solution = result.x.round().astype(int)

# 结果展示
used_food = []
nums = []
for i in range(len(prices)):
    if solution[i] > 0:
        used_food.append(df.iloc[i])
        nums.append(solution[i])

result_df = pd.DataFrame(used_food)
result_df['份数'] = nums

# 打印结果
print('-----------------优化结果如下-----------------')
print(result_df[['餐类', '食物名称', '份数', '价格（元/份）']])
print('总价格:', (result_df['价格（元/份）'] * result_df['份数']).sum())

# 各类食物数量统计
category_counts = result_df['类别'].value_counts()
print(f"总的食物数量: {len(result_df)}")
print("不同类别食物的数量:")
for category, count in zip(category_counts.index, category_counts.values):
    print(f"{category}: {count}")

# 绘制饼图
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('各类食物占比')
plt.show()

# 宏量营养素供能占比
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result_df[col].sum() / result_df[col].sum().sum())

# 绘制宏量营养素供能占比图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.title('宏量营养素供能占比')
plt.show()

# 打印其他信息
print('周总能量摄入量:', result_df[col].sum().sum())
print('每日能量摄入量:', result_df.groupby('星期')['总能量'].sum())
