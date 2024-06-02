import numpy as np  # 导入numpy库，用于进行数值计算
import pandas as pd  # 导入pandas库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
import warnings  # 导入warnings模块，用于忽略警告信息

# 忽略提醒警告，以保持输出整洁
warnings.filterwarnings("ignore")

# 读取调整后的女大学生的一日食谱数据，并使用前向填充处理缺失值
df1 = pd.read_excel('调整后的女大学生的一日食谱.xlsx').fillna(method='ffill')

# 初始化一个空列表，用于存储每种食物的成分含量数据
ls = []
# 读取食品成分表数据
df4 = pd.read_excel('食品成分表.xlsx')
# 获取食品成分表的列名，除了第一列（食品名称）
col = df4.columns[1:]

# 遍历每种食物，计算其成分含量，并将结果存储在列表中
for pca, weight, num in zip(df1['主要成分'], df1['可食部（克/份）'], df1['食用份数']):
    # 根据食品成分表中的数据，计算每种食物的各成分含量
    ls.append((df4[df4.食品 == pca][col].values * weight / 100 * num)[0])

# 将成分含量数据转换为DataFrame
tem = pd.DataFrame(ls, columns=col)
# 将成分含量数据与原始食谱数据合并
result = pd.concat([df1, tem], axis=1)

# 计算食物的能量（卡路里）
col = ['蛋白质', '脂肪', '碳水化合物', '膳食纤维', '酒精（乙醇）']  # 要计算能量的成分列
rios = [4, 9, 4, 2, 7]  # 每种成分的能量转换系数
# 计算每种成分的能量，并添加到结果DataFrame中
tem = result[col] * np.array(rios)
tem.columns = [i + '能量' for i in tem.columns]  # 重命名能量列
result = pd.concat([result, tem], axis=1)  # 将能量数据合并到结果DataFrame中

# 定义食物类别字典，映射食物名称到其所属类别
food_categories = {
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
    '炸鸡块': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '黄花鱼': '畜、禽、鱼、蛋类及制品', '带鱼': '畜、禽、鱼、蛋类及制品',
    '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类', '玉米面': '谷、薯类', '柚子': '蔬菜、菌藻、水果类'
}

# 计算食物类别数量的函数
def count_food_categories(food_list):
    # 初始化一个字典，用于存储每个类别的食物数量
    category_counts = {}
    # 遍历食物列表，根据字典获取食物类别并统计数量
    for food in food_list:
        category = food_categories.get(food, '未知类别')  # 获取食物类别，如果未定义则归类为“未知类别”
        if category in category_counts:
            category_counts[category] += 1  # 如果类别已在字典中，数量加1
        else:
            category_counts[category] = 1  # 如果类别未在字典中，新增类别并设置数量为1
    return len(food_list), category_counts  # 返回总的食物数量和每个类别的数量字典

# 示例食物成分数组，提取结果DataFrame中的主要成分列
food_list = result['主要成分'].unique()

# 计算并输出食物类别数量结果
total_count, category_counts = count_food_categories(food_list)
print(f"总的食物数量: {total_count}")  # 输出总的食物数量
print("不同类别食物的数量:")  # 输出每个类别的食物数量
for category, count in category_counts.items():
    print(f"{category}: {count}")

# 绘制食物类别数量的饼图
labels = category_counts.keys()  # 饼图标签为类别名
sizes = category_counts.values()  # 饼图大小为类别数量
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']  # 定义饼图的颜色
# 设置图表的尺寸为8x8英寸
plt.figure(figsize=(8, 8))

# 绘制饼图，使用不同颜色表示各个食物类别，autopct参数用于显示百分比，shadow参数添加阴影效果，startangle参数设置起始角度为140度
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# 保存饼图到本地文件，文件名为“调整后同学2谷物摄入情况.png”，分辨率为500 DPI
plt.savefig('调整后同学2谷物摄入情况.png', dpi=500)

# 显示饼图
plt.show()

# 计算宏量营养素（蛋白质、脂肪、碳水化合物、膳食纤维、酒精）的供能占比
col = ['蛋白质能量', '脂肪能量', '碳水化合物能量', '膳食纤维能量', '酒精（乙醇）能量']
# 计算每种成分的能量占总能量的比例，并存储到tem变量中
tem = (result[col].sum() / result[col].sum().sum())

# 将宏量营养素供能占比结果保存为Excel文件，文件名为“调整后同学2的日能量贡献.xlsx”
tem.to_excel('调整后同学2的日能量贡献.xlsx')

# 设置条形图的类别（X轴标签）为宏量营养素名称
categories = tem.index
# 设置条形图的数值（Y轴值）为各宏量营养素的能量占比
values = tem.values

# 设置图表的尺寸为10x6英寸
plt.figure(figsize=(10, 6))
# 使用'ggplot'风格
plt.style.use('ggplot')

# 绘制条形图，设置条形的边框颜色为黑色
plt.bar(categories, values, edgecolor='black')

# 设置X轴标签为“摄入能量成分”
plt.xlabel('摄入能量成分')
# 设置Y轴标签为“能量”
plt.ylabel('能量')
# 设置X轴刻度标签的字体大小为12
plt.xticks(fontsize=12)
# 设置Y轴刻度标签的字体大小为12
plt.yticks(fontsize=12)

# 保存条形图到本地文件，文件名为“调整后同学2能量摄入情况.png”，分辨率为500 DPI
plt.savefig('调整后同学2能量摄入情况.png', dpi=500)

# 显示条形图
plt.show()

# 打印输出日总能量摄入量
print('日总能量摄入量', result[col].sum().sum())
# 打印输出宏量营养素供能占比
print(tem)

# 计算非产能主要营养素（钙、铁、锌、维生素A、维生素B1、维生素B2、维生素C）的参考摄入量
col = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
# 计算各非产能主要营养素的总摄入量，并存储到tem变量中
tem = (result[col].sum())

# 将非产能主要营养素参考摄入量结果保存为Excel文件，文件名为“调整后同学2每日膳食非产能主要营养素参考摄入量.xlsx”
tem.to_excel('调整后同学2每日膳食非产能主要营养素参考摄入量.xlsx')

# 设置条形图的类别（X轴标签）为非产能主要营养素名称
categories = tem.index
# 设置条形图的数值（Y轴值）为各非产能主要营养素的摄入量
values = tem.values

# 设置图表的尺寸为10x6英寸
plt.figure(figsize=(10, 6))
# 使用'ggplot'风格
plt.style.use('ggplot')

# 绘制条形图，设置条形的边框颜色为黑色
plt.bar(categories, values, edgecolor='black')

# 设置X轴标签为“摄入能量成分”
plt.xlabel('摄入能量成分')
# 设置Y轴标签为“能量”
plt.ylabel('能量')
# 设置X轴刻度标签的字体大小为12
plt.xticks(fontsize=12)
# 设置Y轴刻度标签的字体大小为12
plt.yticks(fontsize=12)

# 保存条形图到本地文件，文件名为“调整后同学2非产能摄入情况.png”，分辨率为500 DPI
plt.savefig('调整后同学2非产能摄入情况.png', dpi=500)

# 显示条形图
plt.show()

# 打印输出非产能主要营养素参考摄入量
print(tem)

# 计算平均混合食物蛋白质的氨基酸评分（AAS）
# 遍历不同的餐类
for eating in result['餐类'].unique():
    # 获取当前餐类的数据
    tem = result[result.餐类 == eating]
    # 定义需要计算的氨基酸列
    col = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    # 定义氨基酸参考值
    ref = [40, 70, 55, 35, 60, 40, 10, 50]
    # 计算各氨基酸的总摄入量并保存为Excel文件
    (result[col].sum() / result['蛋白质'].sum() * 10).to_excel('同学1氨酸摄入情况.xlsx')

    # 计算氨基酸评分
    score = ((tem[col].sum() / tem['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    # 打印输出当前餐类的AAS评分
    print(f'该同学{eating}的AAS摄入评分为', score)

# 打印输出所有氨基酸的总摄入量
print(result[col].sum() / result['蛋白质'].sum() * 10)
