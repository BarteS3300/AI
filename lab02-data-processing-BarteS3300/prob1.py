import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def readFromCSV(filename):
    df = pd.read_csv(filename)
    return df

def numberOfEmployees(filename):
    df = readFromCSV(filename)
    return df.shape[0]

def numberAndTypesOfInformationForEmployees(filename):
    df = readFromCSV(filename)
    dic = {}
    for index, row in df.iterrows():
        info = ''
        "First Name,Gender,Start Date,Last Login Time,Salary,Bonus %,Senior Management,Team"
        if row['First Name'] != '':
            info += 'First Name, '
        if row['Gender'] != '':
            info += 'Gender, '
        if row['Start Date'] != '':
            info += 'Start Date, '
        if row['Last Login Time'] != '':
            info += 'Last Login Time, '
        if row['Salary'] != '':
            info += 'Salary, '
        if row['Bonus %'] != '':
            info += 'Bonus %, '
        if row['Senior Management'] != '':
            info += 'Senior Management, '
        if row['Team'] != '':
            info += 'Team, '
        if info != '':
            info = info[:-2]
        dic[index] = info
    return dic

def allEmployeesWithFullInformation(filename):
    df = readFromCSV(filename)
    return df.dropna()

def minMeanMaxProperties(filename):
    df = readFromCSV(filename)
    new_df = df.describe()
    new_df = new_df.drop(['count', '25%', '50%', '75%', 'std'])
    new_df = new_df.iloc[[1, 0, 2]]
    return new_df

def allValuesForProperty(filename):
    df = readFromCSV(filename)
    unique_values = {}
    for column in df.columns:
        if df[column].dtype != 'int64' and df[column].dtype != 'float64':
            unique_values[column] = df[column].unique()
    return unique_values

def emptyCellsCorection(filename):
    df = readFromCSV(filename)
    for column in df.columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            df[column].fillna(df[column].mean(), inplace = True)
        else:
            df[column].fillna(df[column].mode()[0], inplace = True)
    return df

def distributionOftheSalaryBySalary(filename):
    df = readFromCSV(filename)
    var_data = df['Salary']
    plt.hist(var_data, bins = 10, color = 'orange', edgecolor = 'gray')
    plt.show()

def distributionOftheSalaryByTeams(filename):
    df = readFromCSV(filename)
    var_data = emptyCellsCorection(filename)
    teams = var_data['Team'].unique()
    salaryByTeam = {}
    for team in teams:
        salaryByTeam[team] = var_data[var_data['Team'] == team]['Salary']
    labels = list(salaryByTeam)
    plt.hist(salaryByTeam.values(), bins = 10, edgecolor = 'gray', label=labels)
    plt.legend(prop = {'size': 6})
    plt.show()
    
def outliers(filename):
    df = readFromCSV(filename)
    
    z = np.abs(stats.zscore(df['Salary']))
    
    threshold = 1.3
    outliers = df[z > threshold]
    
    plt.hist(outliers['Salary'], bins = 10, color = 'orange', edgecolor = 'gray')
    plt.show()

def main():
    filename = 'employees.csv'
    #print("The number of employees is: " + str(numberOfEmployees(filename)))
    #print(numberAndTypesOfInformationForEmployees(filename))
    #print(allEmployeesWithFullInformation(filename))
    #print(minMeanMaxProperties(filename))
    #print(allValuesForProperty(filename))
    #print(emptyCellsCorection(filename))
    #distributionOftheSalaryBySalary(filename)
    distributionOftheSalaryByTeams(filename)
    outliers(filename)
    
main()