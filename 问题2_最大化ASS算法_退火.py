import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# 定义常量和文件路径
DATA_FILE = '问题2\\优化模型数据文件.xlsx'
OUTPUT_DIR = '问题2'
TARGET = 'AA最大化'
SEX = '男'

# 加载和预处理数据的函数
def load_data(file_path):
    df = pd.read_excel(file_path).fillna(method='ffill')
    food_categories = {'大米饭': '谷、薯类', '馒头': '谷、薯类', '花卷': '谷、薯类', '豆浆': '奶、干豆、坚果、种子类及制品', '南瓜粥': '蔬菜、菌藻、水果类', '馄饨': '谷、薯类', '鸡排面': '畜、禽、鱼、蛋类及制品', '馄饨面': '谷、薯类', '红烧牛肉面': '畜、禽、鱼、蛋类及制品', '砂锅面': '谷、薯类', '包子': '谷、薯类', '馅饼': '谷、薯类', '鸡蛋饼': '畜、禽、鱼、蛋类及制品', '土豆丝饼': '谷、薯类', '水煎包': '谷、薯类', '水饺': '谷、薯类', '蒸饺': '谷、薯类', '韭菜盒子': '谷、薯类', '鸡蛋柿子汤': '畜、禽、鱼、蛋类及制品', '萝卜粉丝汤': '蔬菜、菌藻、水果类', '鱼丸汤': '畜、禽、鱼、蛋类及制品', '菠菜汤': '蔬菜、菌藻、水果类', '拌豆腐': '奶、干豆、坚果、种子类及制品', '拌干豆腐': '奶、干豆、坚果、种子类及制品', '拌木耳': '蔬菜、菌藻、水果类', '拌芹菜花生米': '奶、干豆、坚果、种子类及制品', '海带炖白菜': '蔬菜、菌藻、水果类', '白菜炖豆腐': '奶、干豆、坚果、种子类及制品', '鸡肉炖土豆胡萝卜': '畜、禽、鱼、蛋类及制品', '明太鱼炖豆腐': '畜、禽、鱼、蛋类及制品', '炒芹菜粉': '蔬菜、菌藻、水果类', '香菇炒油菜': '蔬菜、菌藻、水果类', '卷心菜炒木耳': '蔬菜、菌藻、水果类', '炒三丝': '蔬菜、菌藻、水果类', '炒豆芽粉条': '蔬菜、菌藻、水果类', '木须柿子': '蔬菜、菌藻、水果类', '木须瓜片': '蔬菜、菌藻、水果类', '地三鲜': '蔬菜、菌藻、水果类', '炒肉扁豆': '畜、禽、鱼、蛋类及制品', '炒肉蒜台': '畜、禽、鱼、蛋类及制品', '炒肉青椒': '畜、禽、鱼、蛋类及制品', '炒肉杏鲍菇': '畜、禽、鱼、蛋类及制品', '炒肉酸菜粉': '畜、禽、鱼、蛋类及制品', '家常豆腐': '奶、干豆、坚果、种子类及制品', '溜肉段': '畜、禽、鱼、蛋类及制品', '锅包肉': '畜、禽、鱼、蛋类及制品', '红烧肉': '畜、禽、鱼、蛋类及制品', '烧排骨': '畜、禽、鱼、蛋类及制品', '宫保鸡丁': '畜、禽、鱼、蛋类及制品', '炸鸡块': '畜、禽、鱼、蛋类及制品', '炒牛肉': '畜、禽、鱼、蛋类及制品', '茄汁沙丁鱼': '畜、禽、鱼、蛋类及制品', '干炸黄花鱼': '畜、禽、鱼、蛋类及制品', '红烧带鱼': '畜、禽、鱼、蛋类及制品', '西瓜': '蔬菜、菌藻、水果类', '香蕉': '蔬菜、菌藻、水果类', '蜜瓜': '蔬菜、菌藻、水果类', '苹果': '蔬菜、菌藻、水果类', '葡萄': '蔬菜、菌藻、水果类', '牛奶': '奶、干豆、坚果、种子类及制品', '酸奶': '奶、干豆、坚果、种子类及制品', '大米粥': '谷、薯类', '小米粥': '谷、薯类', '油条': '谷、薯类', '煮鸡蛋': '畜、禽、鱼、蛋类及制品', '煎鸡蛋': '畜、禽、鱼、蛋类及制品', '蒸地瓜': '谷、薯类', '拌菠菜': '蔬菜、菌藻、水果类', '拌海带丝': '蔬菜、菌藻、水果类', '拌土豆丝': '蔬菜、菌藻、水果类', '橙子': '蔬菜、菌藻、水果类', '炖海带白菜豆腐': '蔬菜、菌藻、水果类', '柚子': '蔬菜、菌藻、水果类'}
    df['类别'] = [food_categories[i] for i in df['食物名称']]
    return df

# 评估当前解决方案的目标函数
def evaluate(df, selection):
    an_sum = df.loc[selection, ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']].sum().sum()
    return an_sum

# 退火算法的实现
def simulated_annealing(df, initial_temp, final_temp, alpha, max_iterations):
    current_solution = random.sample(range(len(df)), 12)
    current_value = evaluate(df, current_solution)
    best_solution = current_solution
    best_value = current_value
    temp = initial_temp
    
    while temp > final_temp:
        for i in range(max_iterations):
            new_solution = current_solution.copy()
            new_solution[random.randint(0, len(new_solution) - 1)] = random.randint(0, len(df) - 1)
            new_value = evaluate(df, new_solution)
            
            delta = new_value - current_value
            if delta > 0 or random.random() < np.exp(delta / temp):
                current_solution = new_solution
                current_value = new_value
                if current_value > best_value:
                    best_solution = current_solution
                    best_value = current_value
        
        temp *= alpha
    
    return best_solution, best_value

# 保存结果的函数
def save_results(selected_foods, df, total_value, output_dir, target, sex):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df = df.iloc[selected_foods]
    result_df['份数'] = 1
    
    result_df[['餐类', '食物名称', '份数', '价格（元/份）']].to_excel(os.path.join(output_dir, f'{target}{sex}同学优化结果.xlsx'))
    
    category_counts = result_df['类别'].value_counts()
    labels = category_counts.index
    sizes = category_counts.values
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.savefig(os.path.join(output_dir, f'{target}{sex}同学谷物摄入情况.png'), dpi=500)
    plt.show()
    
    col_energy = ['蛋白质', '脂肪', '碳水化合物']
    tem = result_df[col_energy].sum() / result_df[col_energy].sum().sum()
    tem.to_excel(os.path.join(output_dir, f'{target}{sex}同学的日能量贡献.xlsx'))
    
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plt.bar(tem.index, tem.values, edgecolor='black')
    plt.xlabel('摄入能量成分')
    plt.ylabel('能量')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(output_dir, f'{target}{sex}同学能量摄入情况.png'), dpi=500)
    plt.show()
    
    col_non_energy = ['钙', '铁', '锌', '维生素A', '维生素B1', '维生素B2', '维生素C']
    tem_non_energy = result_df[col_non_energy].sum()
    tem_non_energy.to_excel(os.path.join(output_dir, f'{target}{sex}同学每日膳食非产能主要营养素参考摄入量.xlsx'))
    
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plt.bar(tem_non_energy.index, tem_non_energy.values, edgecolor='black')
    plt.xlabel('摄入营养成分')
    plt.ylabel('含量')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(output_dir, f'{target}{sex}同学非产能摄入情况.png'), dpi=500)
    plt.show()
    
    col_amino = ['异亮氨酸', '亮氨酸', '赖氨酸', '含硫氨基酸', '芳香族氨基酸', '苏氨酸', '色氨酸', '缬氨酸']
    ref = [40, 70, 55, 35, 60, 40, 10, 50]
    score = ((result_df[col_amino].sum() / result_df['蛋白质'].sum() * 10) / np.array(ref) * 100).sum() / 8
    print(f'{sex}同学的AAS摄入评分为', score)

# 主函数
def main():
    df = load_data(DATA_FILE)
    initial_temp = 1000
    final_temp = 1
    alpha = 0.9
    max_iterations = 1000
    selected_foods, total_value = simulated_annealing(df, initial_temp, final_temp, alpha, max_iterations)
    save_results(selected_foods, df, total_value, OUTPUT_DIR, TARGET, SEX)
    print(f'总氨基酸得分: {total_value}')
    print(f"选择的食品数量：{len(selected_foods)}")

if __name__ == "__main__":
    main()
