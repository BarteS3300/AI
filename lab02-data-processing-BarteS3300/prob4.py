import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from skimage import io
import re

def readFromCSV(filename):
    df = pd.read_csv(filename)
    return df

def emptyCellsCorection(filename):
    df = readFromCSV(filename)
    for column in df.columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            df[column].fillna(df[column].mean(), inplace = True)
        else:
            df[column].fillna(df[column].mode()[0], inplace = True)
    return df

def salaryNormalisation(filename):
    df = emptyCellsCorection(filename)
    salaryClipped = df['Salary'].clip(40000, 140000)
    salaryLog = df['Salary'].apply(lambda x: math.log(x))
    salaryMinMax = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
    m = df['Salary'].mean()
    s = (1 / len(df['Salary']) * sum([(p - m) ** 2 for p in df['Salary']])) ** 0.5
    #std = df['Salary'].std()
    #print('Mean:', m, 'Standard deviation:', s, 'Standard deviation:', std)
    salaryStandardisation = (df['Salary'] - m) / s
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes[0, 0].hist(df['Salary'], bins=10 ,color='red', edgecolor='darkred' , label='Salary')    
    axes[0, 0].set_title('Salary')
    axes[0, 1].hist(salaryClipped, bins=10, color='orange', edgecolor='darkorange', label='Salary Clipped')
    axes[0, 1].set_title('Salary Clipped')
    axes[1, 0].hist(salaryLog, bins=10, color='yellow', edgecolor='orange', label='Salary Log')
    axes[1, 0].set_title('Salary Log')
    axes[1, 1].hist(salaryMinMax, bins=10, color='green', edgecolor='darkgreen', label='Salary MinMax')
    axes[1, 1].set_title('Salary MinMax')
    axes[2, 0].hist(salaryStandardisation, bins=10, color='blue', edgecolor='darkblue', label='Salary Standardisation')
    axes[2, 0].set_title('Salary Standardisation')
    axes[2, 1].set_visible(False)
    fig.tight_layout()
    plt.show()
    
def bonusNormalisation(filename):
    df = emptyCellsCorection(filename)
    bonusClipped = df['Bonus %'].clip(1, 20)
    bonusLog = df['Bonus %'].apply(lambda x: math.log(x))
    bonusMinMax = (df['Bonus %'] - df['Bonus %'].min()) / (df['Bonus %'].max() - df['Bonus %'].min())
    m = df['Bonus %'].mean()
    s = (1 / len(df['Bonus %']) * sum([(p - m) ** 2 for p in df['Bonus %']])) ** 0.5
    bonusStandardisation = (df['Bonus %'] - m) / s
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes[0, 0].hist(df['Bonus %'], bins=10 ,color='red', edgecolor='darkred' , label='Salary')
    axes[0, 0].set_title('Bonus %')
    axes[0, 1].hist(bonusClipped, bins=10, color='orange', edgecolor='darkorange', label='Salary Clipped')
    axes[0, 1].set_title('Bonus % Clipped')
    axes[1, 0].hist(bonusLog, bins=10, color='yellow', edgecolor='orange', label='Salary Log')
    axes[1, 0].set_title('Bonus % Log')
    axes[1, 1].hist(bonusMinMax, bins=10, color='green', edgecolor='darkgreen', label='Salary MinMax')
    axes[1, 1].set_title('Bonus % MinMax')
    axes[2, 0].hist(bonusStandardisation, bins=10, color='blue', edgecolor='darkblue', label='Salary Standardisation')
    axes[2, 0].set_title('Bonus % Standardisation')
    axes[2, 1].set_visible(False)
    fig.tight_layout()
    plt.show()
    
def teamNormalisation(filename):
    df = emptyCellsCorection(filename)
    teams = df['Team'].unique()
    teams = {team: i + 1 for i, team in enumerate(teams)}
    df['Team'] = df['Team'].map(teams)
    labels = ''
    for team in teams:
        print(team)
        labels += str(team) + ' : ' + str(teams[team]) + '\n'
    teamClipped = df['Team'].clip(1, 10)
    teamLog = df['Team'].apply(lambda x: math.log(x))
    teamMinMax = (df['Team'] - df['Team'].min()) / (df['Team'].max() - df['Team'].min())
    m = df['Team'].mean()
    s = (1 / len(df['Team']) * sum([(p - m) ** 2 for p in df['Team']])) ** 0.5
    teamStandardisation = (df['Team'] - m) / s
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes[0, 0].hist(df['Team'], bins=10 ,color='red', edgecolor='darkred', label=labels)
    axes[0, 0].set_title('Team')
    axes[0, 0].legend(prop = {'size': 6})
    axes[0, 1].hist(teamClipped, bins=10, color='orange', edgecolor='darkorange')
    axes[0, 1].set_title('Team Clipped')
    axes[1, 0].hist(teamLog, bins=10, color='yellow', edgecolor='orange')
    axes[1, 0].set_title('Team Log')
    axes[1, 1].hist(teamMinMax, bins=10, color='green', edgecolor='darkgreen')
    axes[1, 1].set_title('Team MinMax')
    axes[2, 0].hist(teamStandardisation, bins=10, color='blue', edgecolor='darkblue')
    axes[2, 0].set_title('Team Standardisation')
    axes[2, 1].set_visible(False)
    fig.tight_layout()
    plt.show()
    
def loadImagesInFolder():
    crtDir = os.getcwd()
    filepath = os.path.join(crtDir, 'images')
    list_files = os.listdir(filepath)
    return list_files

def pixelValuesNormalisation():
    images = loadImagesInFolder()
    df = pd.DataFrame()
    for image in images:
        im = io.imread('images/' + image)
        df = pd.concat([df, pd.DataFrame(im.flatten())])
    df.rename(columns={0: 'Pixel Value'}, inplace=True)
    pixelValuesClipped = df['Pixel Value'].clip(20, 255)
    pixelValuesLog = df['Pixel Value'].apply(lambda x: math.log(x + 1))
    pixelValuesMinMax = (df['Pixel Value'] - df['Pixel Value'].min()) / (df['Pixel Value'].max() - df['Pixel Value'].min())
    m = df['Pixel Value'].mean()
    s = (1 / len(df['Pixel Value']) * sum([(p - m) ** 2 for p in df['Pixel Value']])) ** 0.5
    pixelValuesStandardisation = (df['Pixel Value'] - m) / s
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes[0, 0].hist(df['Pixel Value'], bins=10 ,color='red', edgecolor='darkred')
    axes[0, 0].set_title('Pixel Value')
    axes[0, 1].hist(pixelValuesClipped, bins=10, color='orange', edgecolor='darkorange')
    axes[0, 1].set_title('Pixel Value Clipped')
    axes[1,0].hist(pixelValuesLog, bins=10, color='yellow', edgecolor='orange')
    axes[1,0].set_title('Pixel Value Log')
    axes[1,1].hist(pixelValuesMinMax, bins=10, color='green', edgecolor='darkgreen')
    axes[1,1].set_title('Pixel Value MinMax')
    axes[2,0].hist(pixelValuesStandardisation, bins=10, color='blue', edgecolor='darkblue')
    axes[2,0].set_title('Pixel Value Standardisation')
    axes[2,1].set_visible(False)
    fig.tight_layout()
    plt.show()
    
def readFromTextFile(filename):
    text = ''
    try:
        f = open(filename, 'r', encoding='utf-8')
        text = f.read()
        f.close()
    except IOError:
        print("Fisier corupt!")
    return text

def wordsFromText(filename):
    text = readFromTextFile(filename)
    words = filter(None, re.split(r'[ \n!;:?.,"”]', text))
    return (list(words))

def wordNormalisation(filename):
    words = wordsFromText(filename)
    word_counts = {word: words.count(word) for word in words}
    word_counts = dict(sorted(word_counts.items(), key=lambda x: len(x[0])))
    word_count_normalized = {word: math.log(word_counts[word]**1.8 * len(word)**2) for word in word_counts.keys()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    ax = axes.ravel()
    ax[0].bar(word_counts.keys(), word_counts.values())
    ax[1].bar(word_count_normalized.keys(), word_count_normalized.values())
    plt.show()


def main():
    salaryNormalisation('employees.csv')
    bonusNormalisation('employees.csv')
    teamNormalisation('employees.csv')
    pixelValuesNormalisation()
    wordNormalisation('texts.txt')
    
    
main()
