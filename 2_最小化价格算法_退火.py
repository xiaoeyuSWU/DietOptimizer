import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# 设置 matplotlib 配置
plt.rcParams['axes.unicode_minus'] = False # 解决坐标轴的负号显示问题
warnings.filterwarnings("ignore")

# 加载数据
df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')
food_categories = {'大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品', '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品', '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类', '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品', '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类', '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品', '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品', '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类', '拌海带丝': '蔬菜、菌藻、水果类', '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类', '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'}
df['类别'] = [food_categories[i] for i in df['食物名称']]

# 提取相关数据列用于优化
prices = df['价格（元/份）'].to_list()
calories = df['总能量'].to_list()
protein = df['蛋白质'].to_list()
fat = df['脂肪'].to_list()
carbs = df['碳水化合物'].to_list()
calcium = df['钙'].to_list()
iron = df['铁'].to_list()
zinc = df['锌'].to_list()
vitamin_a = df['维生素A'].to_list()
vitamin_b1 = df['维生素B1'].to_list()
vitamin_b2 = df['维生素B2'].to_list()
vitamin_c = df['维生素C'].to_list()

# 定义目标函数和约束条件
def objective_function(x):
    return sum(prices[i] * x[i] for i in range(len(prices)))

def constraints(x):
    constraints_list = []

    # 每天摄入食物种类大于12种
    constraints_list.append(sum([1 for i in range(len(prices)) if x[i] > 0]) >= 12)

    # 每天摄入的食物要包括五大类别
    categories = df['类别'].unique()
    for category in categories:
        constraints_list.append(sum([x[i] for i in range(len(prices)) if df['类别'][i] == category]) >= 1)

    # 摄入热量为2300-2500之间
    total_calories = sum(calories[i] * x[i] for i in range(len(prices)))
    constraints_list.append(2300 <= total_calories <= 2500)

    # 早餐、午餐、晚餐能量约束
    breakfast_calories = sum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '早餐')
    lunch_calories = sum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '午餐')
    dinner_calories = sum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '晚餐')
    constraints_list.append(690 <= breakfast_calories <= 750)
    constraints_list.append(690 <= lunch_calories <= 1000)
    constraints_list.append(690 <= dinner_calories <= 1000)

    # 微量营养素约束
    constraints_list.append(700 <= sum(calcium[i] * x[i] for i in range(len(prices))) <= 900)
    constraints_list.append(8 <= sum(iron[i] * x[i] for i in range(len(prices))) <= 14)
    constraints_list.append(8 <= sum(zinc[i] * x[i] for i in range(len(prices))) <= 14)
    constraints_list.append(700 <= sum(vitamin_a[i] * x[i] for i in range(len(prices))) <= 900)
    constraints_list.append(1.0 <= sum(vitamin_b1[i] * x[i] for i in range(len(prices))) <= 1.8)
    constraints_list.append(1.0 <= sum(vitamin_b2[i] * x[i] for i in range(len(prices))) <= 1.8)
    constraints_list.append(80 <= sum(vitamin_c[i] * x[i] for i in range(len(prices))) <= 120)

    # 宏量营养素比例约束
    total_energy = [4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i in range(len(prices))]
    total_energy_sum = sum(total_energy[i] * x[i] for i in range(len(prices)))
    constraints_list.append(0.10 * total_energy_sum <= sum(4 * protein[i] * x[i] for i in range(len(prices))) <= 0.15 * total_energy_sum)
    constraints_list.append(0.20 * total_energy_sum <= sum(9 * fat[i] * x[i] for i in range(len(prices))) <= 0.30 * total_energy_sum)
    constraints_list.append(0.50 * total_energy_sum <= sum(4 * carbs[i] * x[i] for i in range(len(prices))) <= 0.65 * total_energy_sum)

    return all(constraints_list)

# 退火算法
def simulated_annealing(objective, constraints, initial_solution, temperature, cooling_rate, iteration_limit):
    current_solution = initial_solution
    current_objective = objective(current_solution)
    best_solution = current_solution
    best_objective = current_objective
    history = []

    for iteration in range(iteration_limit):
        temperature *= cooling_rate
        next_solution = current_solution.copy()
        i = np.random.randint(0, len(next_solution))
        next_solution[i] = np.random.randint(0, 2)  # 随机改变一个食物的选择状态

        if constraints(next_solution):
            next_objective = objective(next_solution)
            delta = next_objective - current_objective
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_solution = next_solution
                current_objective = next_objective

                if current_objective < best_objective:
                    best_solution = current_solution
                    best_objective = current_objective

        history.append(best_objective)

    return best_solution, best_objective, history

# 初始参数设置
initial_solution = [np.random.randint(0, 2) for _ in range(len(prices))]
temperature = 1000
cooling_rate = 0.99
iteration_limit = 10000

# 运行退火算法
best_solution, best_objective, history = simulated_annealing(objective_function, constraints, initial_solution, temperature, cooling_rate, iteration_limit)

# 打印结果
print("选择的食品数量：")
selected_foods = []
selected_food_quantities = []
for i in range(len(prices)):
    if best_solution[i] > 0:
        selected_foods.append(i + 1)
        selected_food_quantities.append(best_solution[i])
        print(f"食品 {i+1}: {best_solution[i]} 份")
print("总成本：", best_objective)

# 生成优化结果
result = []
columns = ['价格（元/份）', '蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C', '异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸', '蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量', '总能量']
for i in selected_foods:
    result.append(df.iloc[i-1].values)
result = pd.DataFrame(result, columns=df.columns)
result['份数'] = selected_food_quantities

print('-----------------优化结果如下-----------------')
print(result[['餐类', '食物名称', '份数', '价格（元/份）']])
print('总价格:', (result['价格（元/份）'] * result['份数']).sum())

# 更新各成分数据
for col in columns:
    result[col] = result[col] * result['份数']

# 保存结果
opt_result = result.copy()
target = '价格最小化'
sex = '男'
category_counts = opt_result['类别'].value_counts()
print(f"总的食物数量: {len(opt_result)}")
print("不同类别食物的数量:")
for category, count in zip(category_counts.index, category_counts.values):
    print(f"{category}: {count}")

result[['餐类', '食物名称', '份数']].to_excel(f'问题2/{target}{sex}同学优化结果.xlsx')

# 绘制饼图
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.savefig(f'问题2/{target}{sex}同学谷物摄入情况.png', dpi=500)
plt.show()

# 宏量营养素供能占比
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result[col].sum() / result[col].sum().sum())
tem.to_excel(f'问题2/{target}{sex}同学的日能量贡献.xlsx')

# 绘制能量摄入条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2/{target}{sex}同学能量摄入情况.png', dpi=500)
plt.show()

print('日总能量摄入量', result[col].sum().sum())
print(tem)

# 非产能主要营养素
col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
tem = (result[col].sum())
tem.to_excel(f'问题2/{target}{sex}同学每日膳食非产能主要营养素参考摄入量.xlsx')

# 绘制非产能摄入条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2/{target}{sex}同学非产能摄入情况.png', dpi=500)
plt.show()

print(tem)

# 计算 AAS 评分
for eating in result['餐类'].unique():
    tem = result[result.餐类 == eating]
    col = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    ref = [40, 70, 55, 35, 60, 40, 10, 50]
    (result[col].sum() / result['蛋白质'].sum() * 10).to_excel(f'问题2/{target}{sex}氨酸摄入情况.xlsx')

    score = ((tem[col].sum() / tem['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    print(f'该同学{eating}的AAS摄入评分为', score)

print((result[col].sum() / result['蛋白质'].sum() * 10))

# 加载数据
df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')
df['类别'] = [food_categories[i] for i in df['食物名称']]

# 提取相关数据列用于优化
prices = df['价格（元/份）'].to_list()
calories = df['总能量'].to_list()
protein = df['蛋白质'].to_list()
fat = df['脂肪'].to_list()
carbs = df['碳水化合物'].to_list()
calcium = df['钙'].to_list()
iron = df['铁'].to_list()
zinc = df['锌'].to_list()
vitamin_a = df['维生素A'].to_list()
vitamin_b1 = df['维生素B1'].to_list()
vitamin_b2 = df['维生素B2'].to_list()
vitamin_c = df['维生素C'].to_list()

# 运行退火算法
best_solution, best_objective, history = simulated_annealing(objective_function, constraints, initial_solution, temperature, cooling_rate, iteration_limit)

# 打印结果
print("选择的食品数量：")
selected_foods = []
selected_food_quantities = []
for i in range(len(prices)):
    if best_solution[i] > 0:
        selected_foods.append(i + 1)
        selected_food_quantities.append(best_solution[i])
        print(f"食品 {i+1}: {best_solution[i]} 份")
print("总成本：", best_objective)

# 生成优化结果
result = []
columns = ['价格（元/份）', '蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C', '异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸', '蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量', '总能量']
for i in selected_foods:
    result.append(df.iloc[i-1].values)
result = pd.DataFrame(result, columns=df.columns)
result['份数'] = selected_food_quantities

print('-----------------优化结果如下-----------------')
display(result[['餐类', '食物名称', '份数', '价格（元/份）']])
print('总价格:', (result['价格（元/份）'] * result['份数']).sum())

# 更新各成分数据
for col in columns:
    result[col] = result[col] * result['份数']

# 保存结果
opt_result = result.copy()
target = '价格最小化'
sex = '女'
category_counts = opt_result['类别'].value_counts()
print(f"总的食物数量: {len(opt_result)}")
print("不同类别食物的数量:")
for category, count in zip(category_counts.index, category_counts.values):
    print(f"{category}: {count}")

result[['餐类', '食物名称', '份数']].to_excel(f'问题2\\{target}{sex}同学优化结果.xlsx')

# 绘制饼图
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.savefig(f'问题2\\{target}{sex}同学谷物摄入情况.png', dpi=500)
plt.show()

# 宏量营养素供能占比
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result[col].sum() / result[col].sum().sum())
tem.to_excel(f'问题2\\{target}{sex}同学的日能量贡献.xlsx')

# 绘制能量摄入条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2\\{target}{sex}同学能量摄入情况.png', dpi=500)
plt.show()

print('日总能量摄入量', result[col].sum().sum())
print(tem)

# 非产能主要营养素
col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
tem = (result[col].sum())
tem.to_excel(f'问题2\\{target}{sex}同学每日膳食非产能主要营养素参考摄入量.xlsx')

# 绘制非产能摄入条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2\\{target}{sex}同学非产能摄入情况.png', dpi=500)
plt.show()

print(tem)

# 计算 AAS 评分
for eating in result['餐类'].unique():
    tem = result[result.餐类 == eating]
    col = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    ref = [40, 70, 55, 35, 60, 40, 10, 50]
    (result[col].sum() / result['蛋白质'].sum() * 10).to_excel(f'问题2\\{target}{sex}氨酸摄入情况.xlsx')

    score = ((tem[col].sum() / tem['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    print(f'该同学{eating}的AAS摄入评分为', score)

print((result[col].sum() / result['蛋白质'].sum() * 10))
