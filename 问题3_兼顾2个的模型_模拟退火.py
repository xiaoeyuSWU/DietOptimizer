import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import random

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

used_food = []

df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')
food_categories = {'大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品', '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品', '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类', '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品', '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类', '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品', '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品', '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类', '拌海带丝': '蔬菜、菌藻、水果类', '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类', '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'}
df['类别'] = [food_categories[i] for i in df['食物名称']]

def calculate_total_nutrition(selected_indices):
    selected_foods = df.iloc[selected_indices]
    total_nutrition = selected_foods[['蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']].sum()
    total_calories = selected_foods['总能量'].sum()
    return total_nutrition, total_calories

def calculate_cost(selected_indices):
    return df.iloc[selected_indices]['价格（元/份）'].sum()

def is_valid_solution(selected_indices):
    if len(selected_indices) < 12:
        return False
    
    selected_foods = df.iloc[selected_indices]
    total_calories = selected_foods['总能量'].sum()
    if total_calories < 2300 or total_calories > 2500:
        return False

    categories = selected_foods['类别'].unique()
    if len(categories) < 5:
        return False

    return True

def simulated_annealing(initial_temp, cooling_rate, num_iterations):
    current_solution = random.sample(range(len(df)), 12)
    while not is_valid_solution(current_solution):
        current_solution = random.sample(range(len(df)), 12)
    
    current_cost = calculate_cost(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp

    for i in range(num_iterations):
        new_solution = current_solution[:]
        idx_to_replace = random.randint(0, len(current_solution) - 1)
        new_idx = random.randint(0, len(df) - 1)
        while new_idx in new_solution:
            new_idx = random.randint(0, len(df) - 1)
        new_solution[idx_to_replace] = new_idx
        
        if is_valid_solution(new_solution):
            new_cost = calculate_cost(new_solution)
            cost_diff = new_cost - current_cost

            if cost_diff < 0 or np.exp(-cost_diff / temperature) > random.random():
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
        
        temperature *= cooling_rate
    
    return best_solution, best_cost

initial_temp = 10000
cooling_rate = 0.995
num_iterations = 10000

best_solution, best_cost = simulated_annealing(initial_temp, cooling_rate, num_iterations)

# 打印结果
print("最佳选择的食品：")
selected_foods = df.iloc[best_solution]
print(selected_foods[['餐类', '食物名称', '价格（元/份）']])
print("总成本：", best_cost)

# 保存结果到Excel
output_path_sa = '问题3\\优化模型数据文件_退火算法.xlsx'
selected_foods.to_excel(output_path_sa, index=False)

print("数据处理完成，并已保存到", output_path_sa)

# 绘制食品类别分布饼图
category_counts = selected_foods['类别'].value_counts()
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("食品类别分布")
plt.savefig('问题3\\退火算法食品类别分布.png', dpi=500)
plt.show()

# 计算总营养成分和能量供给
total_nutrition, total_calories = calculate_total_nutrition(best_solution)
print("总营养成分：")
print(total_nutrition)
print("总能量：", total_calories)

# 宏量营养素供能占比
macro_nutrients = ['蛋白质', '脂肪', '碳水化合物']
macro_nutrient_energies = total_nutrition[macro_nutrients] * [4, 9, 4]
macro_nutrient_energy_ratio = macro_nutrient_energies / total_calories

macro_nutrient_energy_ratio.to_excel('问题3\\退火算法宏量营养素能量占比.xlsx')

# 绘制宏量营养素供能占比条形图
categories = macro_nutrient_energy_ratio.index
values = macro_nutrient_energy_ratio.values

plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('宏量营养素')
plt.ylabel('能量占比')
plt.title('宏量营养素供能占比')
plt.savefig('问题3\\退火算法宏量营养素能量占比.png', dpi=500)
plt.show()
