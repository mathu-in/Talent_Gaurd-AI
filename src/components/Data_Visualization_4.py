import matplotlib.pyplot as plt
import pandas as pd

from config.path import CHART_DIR


def bar_chart(series,title,filename):
    #shape of the chart 10x5
    plt.figure(figsize=(10,10))
    #type of chart
    plt.bar(series.value_counts().index,series.value_counts().values)

    counts = series.value_counts()
    plt.bar(counts.index, counts.values)

    plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    #chart should not come out of the box --- shape
    plt.tight_layout()
    #for saving the chart
    plt.savefig(CHART_DIR / filename)
    #for showing
    # plt.show()
    plt.close()


def histogram(series,title,filename):
    #shape of the chart 10 cross 5
    plt.figure(figsize=(10,10))
    #type of chart
    plt.hist(series,bins=20)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    #chart should not come out of the box --- shape
    plt.tight_layout()
    #for saving the chart
    plt.savefig(CHART_DIR / filename)
    #for showing
    # plt.show()
    plt.close()


def stacked_bar(crosstab,title,filename):
    plt.figure(figsize=(10,5))
    #type of chart
    crosstab.plot(kind='bar',stacked=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename)
    # plt.show()
    plt.close()

def create_visual(data:pd.DataFrame):
    # bar_chart(data['Attrition'],'Attrition Distribution','Attrition_Distribution.png')
    # histogram(data['Age'],'Histogram of Age','Age_Distribution.png')
    # stacked_bar(pd.crosstab(data['OverTime'],data['Attrition']),'OverTime Distribution','OverTime_Distribution.png')
    print("4. Data visualization done --- charts commented")

def main():
    from src.components import load_data
    data = load_data()
    create_visual(data)
    print("Data visualization complete")

if __name__ == '__main__':
    main()