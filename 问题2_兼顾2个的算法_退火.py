import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')

# 食物类别字典
food_categories = {
    '大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品',
    '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品',
    '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类',
    '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品',
    '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类',
    '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类',
    '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品',
    '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类',
    '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类',
    '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品',
    '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类',
    '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类',
    '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类',
    '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品',
    '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品',
    '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品',
    '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品',
    '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品',
    '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品',
    '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品',
    '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品',
    '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类',
    '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类',
    '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品',
    '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品',
    '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类',
    '拌海带丝': '蔬菜、菌藻、水果类', '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类',
    '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'
}
df['类别'] = [food_categories[i] for i in df['食物名称']]

# 提取用于优化的相关数据列
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

an = df[['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']].sum(1).values.tolist()

# 定义约束条件
constraints = {
    "Min Food Types": (lambda x: sum([xi > 0 for xi in x]) >= 12),
    "Category Constraints": (lambda x: all(sum(xi for i, xi in enumerate(x) if df['类别'][i] == category) >= 1 for category in df['类别'].unique())),
    "Min Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x)) >= 2300),
    "Max Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x)) <= 2500),
    "Min Breakfast Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '早餐') >= 690),
    "Max Breakfast Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '早餐') <= 750),
    "Min Lunch Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '午餐') >= 690),
    "Max Lunch Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '午餐') <= 1000),
    "Min Dinner Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '晚餐') >= 690),
    "Max Dinner Calories": (lambda x: sum(calories[i] * xi for i, xi in enumerate(x) if df['餐类'][i] == '晚餐') <= 1000),
    "Min Calcium": (lambda x: sum(calcium[i] * xi for i, xi in enumerate(x)) >= 700),
    "Max Calcium": (lambda x: sum(calcium[i] * xi for i, xi in enumerate(x)) <= 900),
    "Min Iron": (lambda x: sum(iron[i] * xi for i, xi in enumerate(x)) >= 8),
    "Max Iron": (lambda x: sum(iron[i] * xi for i, xi in enumerate(x)) <= 14),
    "Min Zinc": (lambda x: sum(zinc[i] * xi for i, xi in enumerate(x)) >= 8),
    "Max Zinc": (lambda x: sum(zinc[i] * xi for i, xi in enumerate(x)) <= 14),
    "Min Vitamin A": (lambda x: sum(vitamin_a[i] * xi for i, xi in enumerate(x)) >= 700),
    "Max Vitamin A": (lambda x: sum(vitamin_a[i] * xi for i, xi in enumerate(x)) <= 900),
    "Min Vitamin B1": (lambda x: sum(vitamin_b1[i] * xi for i, xi in enumerate(x)) >= 1.0),
    "Max Vitamin B1": (lambda x: sum(vitamin_b1[i] * xi for i, xi in enumerate(x)) <= 1.8),
    "Min Vitamin B2": (lambda x: sum(vitamin_b2[i] * xi for i, xi in enumerate(x)) >= 1.0),
    "Max Vitamin B2": (lambda x: sum(vitamin_b2[i] * xi for i, xi in enumerate(x)) <= 1.8),
    "Min Vitamin C": (lambda x: sum(vitamin_c[i] * xi for i, xi in enumerate(x)) >= 80),
    "Max Vitamin C": (lambda x: sum(vitamin_c[i] * xi for i, xi in enumerate(x)) <= 120),
    "Min Protein": (lambda x: sum(4 * protein[i] * xi for i, xi in enumerate(x)) >= 0.10 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
    "Max Protein": (lambda x: sum(4 * protein[i] * xi for i, xi in enumerate(x)) <= 0.15 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
    "Min Fat": (lambda x: sum(9 * fat[i] * xi for i, xi in enumerate(x)) >= 0.20 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
    "Max Fat": (lambda x: sum(9 * fat[i] * xi for i, xi in enumerate(x)) <= 0.30 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
    "Min Carbs": (lambda x: sum(4 * carbs[i] * xi for i, xi in enumerate(x)) >= 0.50 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
    "Max Carbs": (lambda x: sum(4 * carbs[i] * xi for i, xi in enumerate(x)) <= 0.65 * sum(4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i, xi in enumerate(x))),
}

# 目标函数
def objective_function(x):
    return 20 * sum(prices[i] * xi for i, xi in enumerate(x)) - sum(an[i] * xi for i, xi in enumerate(x))

# 退火算法
def simulated_annealing(objective, constraints, bounds, maxiter=1000, initial_temp=100, cooling_rate=0.95):
    def accept_prob(old_cost, new_cost, temp):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temp)
    
    # 初始解
    current_solution = [np.random.randint(low=0, high=2) for _ in range(len(prices))]
    current_cost = objective(current_solution)
    
    # 退火过程
    temp = initial_temp
    best_solution = current_solution
    best_cost = current_cost
    
    for i in range(maxiter):
        # 生成邻域解
        new_solution = current_solution[:]
        index = np.random.randint(0, len(new_solution))
        new_solution[index] = 1 - new_solution[index]  # 翻转一个随机选择的位
        
        # 检查新解是否满足所有约束
        if all(constraint(new_solution) for constraint in constraints.values()):
            new_cost = objective(new_solution)
            if accept_prob(current_cost, new_cost, temp) > np.random.rand():
                current_solution = new_solution
                current_cost = new_cost
                
                # 更新最佳解
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
        
        # 降温
        temp *= cooling_rate
    
    return best_solution, best_cost

# 定义变量边界
bounds = [(0, 1) for _ in range(len(prices))]

# 执行退火算法
best_solution, best_cost = simulated_annealing(objective_function, constraints, bounds)

# 输出结果
selected_foods = [i for i, xi in enumerate(best_solution) if xi > 0]
print("选择的食品数量：")
for i in selected_foods:
    print(f"食品 {i+1}: {best_solution[i]} 份")
print("总成本：", best_cost)

# 生成优化结果的 DataFrame
result = []
num = []
for i in selected_foods:
    result.append(df.iloc[i].values)
    num.append(best_solution[i])
result = pd.DataFrame(result, columns=df.columns)
result['份数'] = num

print('-----------------优化结果如下-----------------')
print(result[['餐类', '食物名称', '份数', '价格（元/份）']])
print('总价格:', (result['价格（元/份）'] * result['份数']).sum())

# 保存结果到 Excel 文件
result[['餐类', '食物名称', '份数']].to_excel(f'问题2\\退火算法优化结果.xlsx')

# 生成类别分布的饼图
category_counts = result['类别'].value_counts()
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.savefig(f'问题2\\退火算法类别分布.png', dpi=500)
plt.show()

# 生成宏量营养素能量贡献的柱状图
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result[col].sum() / result[col].sum().sum())
tem.to_excel(f'问题2\\退火算法能量贡献.xlsx')

categories = tem.index
values = tem.values

plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2\\退火算法能量摄入.png', dpi=500)
plt.show()

print('日总能量摄入量', result[col].sum().sum())
print(tem)

col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
tem = result[col].sum()
tem.to_excel(f'问题2\\退火算法膳食非产能营养素.xlsx')

categories = tem.index
values = tem.values

plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入营养成分')
plt.ylabel('含量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2\\退火算法非产能摄入.png', dpi=500)
plt.show()

print(tem)
