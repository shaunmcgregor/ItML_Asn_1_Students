import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets
import IPython.display

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList. If no values are set some `fullEDA` tabs will display no information
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList. If no values are set some `fullEDA` tabs will display no information.
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 
    
    countValues(self, columnToCount)
        Provides value conuts for the column passed in

         Parameters
        ----------
        columnToCount : string
            index of the column to display the value counts
    
    showHeatmap(self)
        Displays a correlation heatmap for all columns in self.data

    imbalanceCheck(self, tooManyCount = 20)
        Will show the value_counts of the target to help discover any potential imbalances.

         Parameters
        ----------
        tooManyCount : Int
            The number of unique values before the function will not show the value counts as the target is assumed to be numeric. If this is set to 0, the value counts will be shown.

    displayValueCounts(self)
        Displays interactive widget that allows te user to select a categorical feature to display the value counts

    sampleData(self)
        Allows a user to select the number of rows of sample data to display

    fullEDA(includePairPlot=False, includeCorrelationHeatmap=False)
        Displays the full EDA process. 

         Parameters
        ----------
        includePairPlot : bool
            If true the pairplot tab will be run and a pairplot will be shown. WARNING: this process can be extremely slow
        includeCorrelationHeatmap : bool
            If true the correlation tab will be run a correlation heatmap will be shown. WARNING: this process can be extremely slow
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def describe(self):
        return self.data.describe()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure
    
    def countValues(self, columnToCount):
        print(self.data[columnToCount].value_counts(),"\n")

    def showHeatmap(self):
        correlations = self.data.corr()
        sns.heatmap(correlations, center=0, linewidths=.5, annot=True, cmap="YlGnBu", yticklabels=True)
    
    def imbalanceCheck(self, tooManyCount = 20):
        if len(self.data[self.target].value_counts()) > tooManyCount & tooManyCount != 0: 
            print("Unique target value counts greater than",tooManyCount,"target is likely not catergorical\n","If you wish to see all target value_counts() enter 0 as tooManyCount")
        else:
            print(self.data[self.target].value_counts())
            
    def displayValueCounts(self):
        if len(self.cat) > 0:
            catListSelectionWidget = widgets.Dropdown(options=self.cat, description='Category Field:', disabled=False, value=self.cat[0])
            countsOutput = widgets.Output()
            display(catListSelectionWidget, countsOutput)
            with countsOutput:
                print(self.countValues(self.cat[0]))
            def onCatSelectionChanged(change):
                countsOutput.clear_output()
                with countsOutput:
                    display(self.countValues(change['new']))
            catListSelectionWidget.observe(onCatSelectionChanged, names='value')
        else:
            display(print("No catergorical features set, set catergorical features using `edaDF.setCat()`"))

    def sampleData(self):
        amountSlider = widgets.IntSlider(min=1, max=50, description='Number of Records:', value=10)
        sampleOutput = widgets.Output()
        display(amountSlider, sampleOutput)
        with sampleOutput:
            print(self.data.sample(10))
        def onSliderChange(change):
            sampleOutput.clear_output()
            with sampleOutput:
                print(self.data.sample(change['new']))
        amountSlider.observe(onSliderChange, names='value')
    
    def fullEDA(self, includePairPlot = False, includeCorrelationHeatmap = False):
        generalInfo = (widgets.Output(), 'General Info')
        catergorical = (widgets.Output(), 'Categorical')
        numerical = (widgets.Output(), 'Numerical')
        pairPlot = (widgets.Output(), 'Pair Plot')
        describe = (widgets.Output(), 'Describe')
        heatMap = (widgets.Output(), 'Correlations')
        missingValues = (widgets.Output(), 'Missing Values')
        sampleData = (widgets.Output(), 'Sample')
        imbalanceCheck = (widgets.Output(), 'Check Imbalnce')
        valueCounts = (widgets.Output(), 'Value Counts')

        #you can control the order the tabs appear by changing the order of this list
        childs = [generalInfo, describe, catergorical, numerical, valueCounts, missingValues, sampleData,imbalanceCheck]
        if includePairPlot:
            childs.append(pairPlot)
        if includeCorrelationHeatmap:
            childs.append(heatMap)
        
        #get the Tab children from the childs list
        tab = widgets.Tab(children = [child[0] for child in childs])
        #set the titles based on the second part of the the tuples
        for child in childs:
            tab.set_title(childs.index(child), child[1])

        display(tab)

        with generalInfo[0]:
            self.info()

        with describe[0]:
            display(self.describe())

        with catergorical[0]:
            if len(self.cat) > 0:
                fig2 = self.countPlots(splitTarg=True, show=False)
                plt.show(fig2)
            else:
                display(print("No catergorical features set, set catergorical features using `edaDF.setCat()`"))
        
        with numerical[0]:
            if len(self.num) > 0:
                fig3 = self.histPlots(kde=True, show=False)
                plt.show(fig3)
            else:
                display(print("No numerical features set, set numerical features using `edaDF.setNum()`"))

        with valueCounts[0]:
            self.displayValueCounts()

        if includeCorrelationHeatmap:
            with heatMap[0]:
                corrHeatmap = self.showHeatmap()
                plt.show(corrHeatmap)

        if includePairPlot:
            with pairPlot[0]:
                pairFigure =  sns.pairplot(self.data)
                plt.show(pairFigure)
        
        with missingValues[0]:
            display(print("Number of rows missing values by column, sorted by amount decending\n", self.data.isnull().sum().sort_values(ascending=False)))

        with sampleData[0]:
            self.sampleData()

        with imbalanceCheck[0]:
            self.imbalanceCheck()


        #OutLiers !!
        #only run if num and cat are not empty -/
        #sample Data - -/
        #finish missing values  - Huh
        #covariants - 
        #imbalance -/

