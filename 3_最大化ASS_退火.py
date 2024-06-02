import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 忽略提醒
import warnings
warnings.filterwarnings("ignore")

# 设置绘图参数
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴的负号显示问题

# 读取数据
df1 = pd.read_excel('附件3：某高校学生食堂一日三餐主要食物信息统计表.xlsx').fillna(method='ffill')

# 获取成分含量文件列表并合并成分含量数据
a = os.listdir('成分含量')[:4]
df2 = pd.DataFrame(columns=['食品'])
for i in a:
    df2 = pd.merge(df2, pd.read_excel(f'成分含量/{i}'), on='食品', how='right')

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
output_path = '问题3/优化模型数据文件.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
total.to_excel(output_path)

# 读取数据
df = pd.read_excel('问题3/优化模型数据文件.xlsx').fillna(method='ffill')

# 食物分类
food_categories = {'大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品', '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品', '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类', '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品', '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '鸡蛋': '畜、禽、鱼、蛋类及制品', '牛奶': '奶、干豆、坚果、种子类及制品', '紫米面包': '谷、薯类', '鸡蛋灌饼': '谷、薯类', '海带绿豆汤': '蔬菜、菌藻、水果类', '小葱拌豆腐': '奶、干豆、坚果、种子类及制品'}

# 营养素目标范围
nutrient_targets = {
    '总能量': (2300, 2500),
    '蛋白质': (70, 90),
    '脂肪': (50, 80),
    '碳水化合物': (300, 400),
    '钙': (800, 1000),
    '铁': (10, 15),
    '锌': (12, 15),
    '维生素A': (700, 1000),
    '维生素B1': (1.1, 1.5),
    '维生素B2': (1.2, 1.6),
    '维生素C': (85, 100)
}

# 初始化食物选择
food_choices = list(df['食物名称'].unique())
num_foods = len(food_choices)

# 退火算法参数
initial_temp = 1000
final_temp = 1
alpha = 0.99
max_iter = 1000

# 定义目标函数
def objective_function(food_indices):
    selected_foods = df.iloc[food_indices]
    total_nutrients = selected_foods[['蛋白质', '脂肪', '碳水化合物', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']].sum()
    total_energy = selected_foods['总能量'].sum()

    # 计算各营养素与目标范围的偏离度
    penalty = 0
    for nutrient, (lower, upper) in nutrient_targets.items():
        nutrient_value = total_nutrients[nutrient] if nutrient != '总能量' else total_energy
        if nutrient_value < lower:
            penalty += (lower - nutrient_value) ** 2
        elif nutrient_value > upper:
            penalty += (nutrient_value - upper) ** 2

    return penalty

# 退火算法
def simulated_annealing():
    current_solution = np.random.choice(num_foods, size=20, replace=False)
    current_objective = objective_function(current_solution)
    best_solution = current_solution.copy()
    best_objective = current_objective

    temp = initial_temp
    while temp > final_temp:
        for _ in range(max_iter):
            new_solution = current_solution.copy()
            swap_idx = np.random.choice(20, size=2, replace=False)
            new_solution[swap_idx[0]], new_solution[swap_idx[1]] = new_solution[swap_idx[1]], new_solution[swap_idx[0]]
            new_objective = objective_function(new_solution)
            if new_objective < current_objective or np.random.rand() < np.exp((current_objective - new_objective) / temp):
                current_solution = new_solution
                current_objective = new_objective
                if current_objective < best_objective:
                    best_solution = current_solution
                    best_objective = current_objective
        temp *= alpha

    return best_solution, best_objective

# 运行退火算法
best_solution, best_objective = simulated_annealing()

# 输出最佳食物选择和对应的营养成分
best_foods = df.iloc[best_solution]
print("最佳食物选择:")
print(best_foods[['食物名称', '总能量', '蛋白质', '脂肪', '碳水化合物', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']])

# 保存最佳食物选择到Excel
output_path = '问题3/退火算法优化结果.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
best_foods.to_excel(output_path, index=False)

# 绘制食物类别饼图
category_counts = best_foods['食物名称'].map(food_categories).value_counts()
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("最佳食物选择类别分布")
plt.ylabel('')
plt.show()

# 绘制营养成分能量贡献条形图
nutrient_energy_contribution = best_foods[['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']].sum()
nutrient_energy_contribution.plot(kind='bar')
plt.title("营养成分能量贡献")
plt.xlabel("营养成分")
plt.ylabel("能量（千卡）")
plt.show()
