import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger

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
food_categories = {'大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品', '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品', '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类', '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品', '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类', '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品', '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品', '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类', '拌海带丝': '蔬菜、菌藻、水果类', '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类', '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'}
df['类别'] = [food_categories[i] for i in df['食物名称']]

# 创建优化模型
def create_and_solve_model(df, target='AAS最大化', sex='男'):
    lss = []
    nums = []
    day = []
    for o in range(1, 8):
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
        an = df[['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']].sum(1).values.tolist()

        # 创建优化问题
        model = LpProblem(name="diet-problem", sense=LpMinimize)

        # 定义决策变量
        x = [LpVariable(f"x{i}", cat=LpInteger, lowBound=0, upBound=2) for i in range(len(prices))]

        # 添加目标函数
        model += lpSum(-an[i] * x[i] for i in range(len(prices))), "Total Cost"

        # 添加约束条件
        model += lpSum([x[i] != 0 for i in range(len(prices))]) >= 12, "Min Food Types"

        categories = df['类别'].unique()
        for category in categories:
            model += lpSum([x[i] for i in range(len(prices)) if df['类别'][i] == category]) >= 1, f"Category {category} Constraint"

        model += lpSum(calories[i] * x[i] for i in range(len(prices))) >= 2300, "Min Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices))) <= 2500, "Max Calories"

        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '早餐') >= 590, "Min Breakfast Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '早餐') <= 850, "Max Breakfast Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '午餐') >= 590, "Min Lunch Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '午餐') <= 1100, "Max Lunch Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '晚餐') >= 590, "Min Dinner Calories"
        model += lpSum(calories[i] * x[i] for i in range(len(prices)) if df['餐类'][i] == '晚餐') <= 1100, "Max Dinner Calories"

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

        # 求解模型
        model.solve()

        drop_index = []
        ls = []
        num = []

        # 打印结果
        for i in range(len(prices)):
            if x[i].varValue > 0:
                ls.append(i+1)
                drop_index.append(i)
                num.append(round(x[i].varValue))
                lss.append(df.iloc[[i]].index[0])
                nums.append(round(x[i].varValue))
                day.append(o)

        drop_indexx = []
        drop_indexx.append(drop_index[0])
        drop_indexx.append(drop_index[-1])
        drop_indexx.append(drop_index[7])

        df = df.drop(df.iloc[drop_indexx].index, axis=0)

    return lss, nums, day

# 调用函数
lss, nums, day = create_and_solve_model(df)

# 构建结果数据框
result = []
col = ['价格（元/份）', '蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）', '钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C', '异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸', '蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量', '总能量']
for i in lss:
    result.append(df.iloc[i].values)
result = pd.DataFrame(result, columns=df.columns)
result['份数'] = nums
result['星期'] = day

# 输出优化结果
output_dir = '问题3'
os.makedirs(output_dir, exist_ok=True)
result[['餐类', '食物名称', '份数']].to_excel(f'{output_dir}/AAS最大化男同学优化结果.xlsx')

# 绘制饼图
category_counts = result['类别'].value_counts()
labels = category_counts.index
sizes = category_counts.values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.savefig(f'{output_dir}/AAS最大化男同学谷物摄入情况.png', dpi=500)
plt.show()

# 宏量营养素供能占比
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
tem = (result[col].sum() / result[col].sum().sum())
tem.to_excel(f'{output_dir}/AAS最大化男同学的日能量贡献.xlsx')

# 绘制能量贡献条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'{output_dir}/AAS最大化男同学能量摄入情况.png', dpi=500)
plt.show()

print('周总能量摄入量', result[col].sum().sum())
print('每日能量摄入量', result.groupby('星期')['总能量'].sum())

# 非产能营养素摄入情况
col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
tem = (result[col].sum())
tem.to_excel(f'{output_dir}/AAS最大化男同学每日膳食非产能主要营养素参考摄入量.xlsx')

# 绘制非产能营养素条形图
categories = tem.index
values = tem.values
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.bar(categories, values, edgecolor='black')
plt.xlabel('摄入能量成分')
plt.ylabel('能量')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'{output_dir}/AAS最大化男同学非产能摄入情况.png', dpi=500)
plt.show()

# 计算并输出每餐AAS评分
col = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
ref = [40, 70, 55, 35, 60, 40, 10, 50]
for eating in result['餐类'].unique():
    tem = result[result.餐类 == eating]
    score = ((tem[col].sum() / tem['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    print(f'该同学{eating}的AAS摄入评分为', score)
(result[col].sum() / result['蛋白质'].sum() * 10).to_excel(f'{output_dir}/AAS最大化男同学氨酸摄入情况.xlsx')
