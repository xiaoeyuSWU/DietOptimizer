import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpInteger, PULP_CBC_CMD

# 加载数据
df = pd.read_excel('问题2\\优化模型数据文件.xlsx').fillna(method='ffill')
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

# 提取相关数据
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

# 模型
model = LpProblem(name="diet-problem", sense=LpMinimize)

# 定义决策变量
x = [LpVariable(f"x{i}", cat=LpInteger, lowBound=0) for i in range(len(prices))]

# 增加目标函数
model += lpSum(20*prices[i] * x[i] - an[i] * x[i] for i in range(len(prices))), "Total Cost"

model += lpSum([x[i] != 0 for i in range(len(prices))]) >= 12, "Min Food Types"

categories = df['类别'].unique()
for category in categories:
    model += lpSum([x[i] for i in range(len(prices)) if df['类别'][i] == category]) >= 1, f"Category {category} Constraint"

model += lpSum(calories[i] * x[i] for i in range(len(prices))) >= 2300, "Min Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices))) <= 2500, "Max Calories"

model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '早餐') >= 690, "Min Breakfast Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '早餐') <= 750, "Max Breakfast Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '午餐') >= 690, "Min Lunch Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '午餐') <= 1000, "Max Lunch Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '晚餐') >= 690, "Min Dinner Calories"
model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '晚餐') <= 1000, "Max Dinner Calories"

model += lpSum(calcium[i] * x[i] for i in range(len(prices))) >= 700, "Min Calcium"
model += lpSum(calcium[i] * x[i] for i in range(len(prices))) <= 900, "Max Calcium"
model += lpSum(iron[i] * x[i] for i in range(len(prices))) >= 8, "Min Iron"
model += lpSum(iron[i] * x[i] for i in range(len(prices))) <= 14, "Max Iron"
model += lpSum(zinc[i] * x[i] for i in range(len(prices))) >= 8, "Min Zinc"
model += lpSum(zinc[i] * x[i] for i in range(len(prices))) <= 14, "Max Zinc"
model += lpSum(vitamin_a[i] * x[i] for i in range(len(prices))) >= 700, "Min Vitamin A"
model += lpSum(vitamin_a[i] * x[i] for i in range(len(prices))) <= 900, "Max Vitamin A"
model += lpSum(vitamin_b1[i] * x[i] for i in range(len(prices))) >= 1.0, "Min Vitamin B1"
model += lpSum(vitamin_b1[i] * x[i] for i in range(len(prices))) <= 1.8, "Max Vitamin B1"
model += lpSum(vitamin_b2[i] * x[i] for i in range(len(prices))) >= 1.0, "Min Vitamin B2"
model += lpSum(vitamin_b2[i] * x[i] for i in range(len(prices))) <= 1.8, "Max Vitamin B2"
model += lpSum(vitamin_c[i] * x[i] for i in range(len(prices))) >= 80, "Min Vitamin C"
model += lpSum(vitamin_c[i] * x[i] for i in range(len(prices))) <= 120, "Max Vitamin C"

total_energy = [4 * protein[i] + 9 * fat[i] + 4 * carbs[i] for i in range(len(prices))]
model += lpSum(4 * protein[i] * x[i] for i in range(len(prices))) >= 0.10 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Min Protein"
model += lpSum(4 * protein[i] * x[i] for i in range(len(prices))) <= 0.15 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Max Protein"
model += lpSum(9 * fat[i] * x[i] for i in range(len(prices))) >= 0.20 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Min Fat"
model += lpSum(9 * fat[i] * x[i] for i in range(len(prices))) <= 0.30 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Max Fat"
model += lpSum(4 * carbs[i] * x[i] for i in range(len(prices))) >= 0.50 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Min Carbs"
model += lpSum(4 * carbs[i] * x[i] for i in range(len(prices))) <= 0.65 * lpSum(total_energy[i] * x[i] for i in range(len(prices))), "Max Carbs"

# 运行模型
model.solve()

ls = []
num = []
print("选择的食品数量：")
for i in range(len(prices)):
    if x[i].varValue > 0:
        ls.append(i+1)
        num.append(x[i].varValue)
        print(f"食品 {i+1}: {x[i].varValue} 份")
print("总成本：", sum(prices[i] * x[i].varValue for i in range(len(prices))))

result = []
col = ['价格（元/份）', '蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C', '异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸', '蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量', '总能量']
for i in ls:
    result.append(df.iloc[i-1].values)
result = pd.DataFrame(result, columns=df.columns)
result['份数'] = num

print('-----------------优化结果如下-----------------')
print(result[['餐类', '食物名称', '份数', '价格（元/份）']])
print('总价格:', (result['价格（元/份）'] * result['份数']).sum())
for i in col:
    result[i] = result[i] * result['份数']

opt_result = result.copy()

target = 'AAS价格均衡'
sex = '男'
category_counts = opt_result['类别'].value_counts()
print(f"总的食物数量: {len(opt_result)}")
print("不同类别食物的数量:")
for category, count in zip(category_counts.index, category_counts.values):
    print(f"{category}: {count}")

result[['餐类', '食物名称', '份数']].to_excel(f'问题2\\{target}{sex}同学优化结果.xlsx')

labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.savefig(f'问题2\\{target}{sex}同学谷物摄入情况.png', dpi=500)
plt.show()

col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result[col].sum() / result[col].sum().sum())
tem.to_excel(f'问题2\\{target}{sex}同学的日能量贡献.xlsx')

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

col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
tem = result[col].sum()
tem.to_excel(f'问题2\\{target}{sex}同学每日膳食非产能主要营养素参考摄入量.xlsx')

categories = tem.index
values = tem.values

plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入营养成分')
plt.ylabel('含量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'问题2\\{target}{sex}同学非产能摄入情况.png', dpi=500)
plt.show()

print(tem)

for eating in result['餐类'].unique():
    tem = result[result.餐类 == eating]
    col = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    ref = [40, 70, 55, 35, 60, 40, 10, 50]
    (result[col].sum() / result['蛋白质'].sum() * 10).to_excel(f'问题2\\{target}{sex}氨酸摄入情况.xlsx')
    score = ((tem[col].sum() / tem['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    print(f'该同学{eating}的AAS摄入评分为', score)

print(result[col].sum() / result['蛋白质'].sum() * 10)
