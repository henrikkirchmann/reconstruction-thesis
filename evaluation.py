import copy
import os
import itertools
from collections import Counter as mset
from statistics import median
#from pathlib import Path
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pm4py
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
from pathlib import Path
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.process_tree import semantics
from pm4py.statistics.variants.log import get as variants_module

from eventFrequencyPT import eventFreqPT

from generateLogFreq import GenerationTree

from generateLogUniform import generateLog as generateLogUniform
from generateLogStaticDistribution import generateLog as generateLogStaticDistribution
from generateLogDynamicDistribution import generateLog as generateLogDynamicDistribution
from generateLogFreq import generateLog as generateLogFreq
from generateLogMA import generateLog as generateLogMA


def getLengthOfLongestTrace(log):
    trace_lengths = [len(trace) for trace in log]
    return max(trace_lengths)


def transformTraceToString(generatedTrace):
    traceString = ""
    for event in generatedTrace._list:
        if event == generatedTrace._list[-1]:
            traceString += event._dict.get('concept:name')
        else:
            traceString += event._dict.get('concept:name') + ", "
    return traceString


def transformLogToTraceStringList(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        for event in trace._list:
            log_list[i].append(event._dict.get('concept:name'))
        i += 1
    return log_list


def transformLogInStringList(log):
    stringList = list()
    for trace in log:
        traceString = ""
        for event in trace:
            traceString += (" " + event)
        stringList.append(traceString)
    return stringList


def getLogs(processTree, numberOfLogsToGenerate, numberOfCasesInOriginalLog, strategy, im, fm, originalLog, variance, maxTraceLength):
    # list of all Eventlog() 's generated
    generatedEventLogList = list()
    # list of Logs that have their traces as strings
    generatedLogList = list()
    print('######################')
    print(strategy)
    print('######################')
    for i in range(numberOfLogsToGenerate):
        processTreeCopy = copy.deepcopy(processTree)
        if strategy == "A":
            log, eventlog = generateLogUniform(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "B":
            log, eventlog = generateLogStaticDistribution(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "C":
            log, eventlog = generateLogDynamicDistribution(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == ("D with Variance " + str(variance)):
            log, eventlog = generateLogFreq(processTreeCopy, numberOfCasesInOriginalLog, variance)
        elif strategy == citationMA:
            log, eventlog = generateLogMA(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "PM4Py basic playout of a Petri net":
            eventlog = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
                simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: numberOfCasesInOriginalLog})
            log = transformLogToTraceStringList(eventlog)
        elif strategy == "PM4Py basic playout of a process tree":
            eventlog = semantics.generate_log(processTreeCopy, no_traces=numberOfCasesInOriginalLog)
            log = transformLogToTraceStringList(eventlog)
        elif strategy == "PM4Py's Stochastic Play-Out\nof a Petri Net":
            eventlog = simulator.apply(net, im, fm, parameters={
                simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: originalLog,
                simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.NO_TRACES: numberOfCasesInOriginalLog,
                simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.MAX_TRACE_LENGTH: maxTraceLength},
                                       variant=simulator.Variants.STOCHASTIC_PLAYOUT)
            log = transformLogToTraceStringList(eventlog)
        generatedLogList.append(log)
        generatedEventLogList.append(eventlog)
    return generatedLogList, generatedEventLogList


def getMinOfList(logList, numberOfLogsToEvaluate):
    return min(logList[:numberOfLogsToEvaluate])


def getMaxOfList(logList, numberOfLogsToEvaluate):
    return max(logList[:numberOfLogsToEvaluate])


def getMedianOfList(logList, numberOfLogsToEvaluate):
    return median(logList[:numberOfLogsToEvaluate])


def getAvgOfList(logList, numberOfLogsToEvaluate):
    numberOfTraceVariants = 0
    for i in range(numberOfLogsToEvaluate):
        numberOfTraceVariants += logList[i]
    avgNumberOfTraceVariants = numberOfTraceVariants / numberOfLogsToEvaluate
    return avgNumberOfTraceVariants


def getMinOfListList(logLists, numberOfLogsToEvaluate):
    minLengths = list()
    for i in range(numberOfLogsToEvaluate):
        minLengths.append(min(logLists[i]))
    return (min(minLengths))


def getMaxFromListList(logLists, numberOfLogsToEvaluate):
    maxLengths = list()
    for i in range(numberOfLogsToEvaluate):
        maxLengths.append(max(logLists[i]))
    return (max(maxLengths))


def getAverageFromListList(logLists, numberOfLogsToEvaluate):
    logSize = len(logLists[0])
    traceLengthLogs = list()
    for i in range(numberOfLogsToEvaluate):
        traceLengthLogs.append(sum(logLists[i]) / logSize)
    avgNumberOfTraceLength = sum(traceLengthLogs) / numberOfLogsToEvaluate
    return avgNumberOfTraceLength


def getAverageMediansFromListList(logLists, numberOfLogsToEvaluate):
    traceLengthLogs = list()
    for i in range(numberOfLogsToEvaluate):
        traceLengthLogs.append(median(logLists[i]))
    avgNumberOfTraceLength = sum(traceLengthLogs) / numberOfLogsToEvaluate
    return avgNumberOfTraceLength


def getNumberOfTraceVariantsList(logList):
    traceVariantList = list()
    for log in logList:
        traceVariantList.append(len([list(x) for x in set(tuple(x) for x in log)]))
    return traceVariantList


def getTraceLengthsList(logList):
    logTraceLengths = list()
    for log in logList:
        traceLengths = list()
        for trace in log:
            traceLengths.append(len(trace))
        logTraceLengths.append(traceLengths)
    return logTraceLengths


def getMultiSetIntersection(generatedLogList, originalLogList):
    multiSetIntersectionSizeList = list()
    for generatedLog in generatedLogList:
        intersection = mset(transformLogInStringList(originalLogList)) & mset(transformLogInStringList(generatedLog))
        print("Intersection:")
        print(intersection)
        multiSetIntersectionSizeList.append(len(list(intersection.elements())))
    return multiSetIntersectionSizeList


def getEMD(generatedEventLogList, originalEventLog):
    emdList = list()
    for log in generatedEventLogList:
        originalLogLanguage = variants_module.get_language(originalEventLog)
        generatedLogLanguage = variants_module.get_language(log)
        emd = emd_evaluator.apply(generatedLogLanguage, originalLogLanguage)
        emdList.append(emd)
    return emdList


def set_axis_style(ax, labels, variance):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    xticks = plt.xticks()[1]
    plt.setp(xticks[:], rotation=30, ha='right')
    ax.set_xlabel('Play-Out Strategy', labelpad=20)


def transfromTraceLengthsToDataframes(originalTL: list[list], generatedTLs: list[list], numberOfLogs: list,
                                      strategies: list, logname):
    originalTLscaledToNumberOfLogs = []
    for i in range(numberOfLogs[-1]):
        originalTLscaledToNumberOfLogs.extend(originalTL[0])
    traceLengths = []
    for logs in generatedTLs:
        traceLengths.extend(logs)
        traceLengths.extend(originalTLscaledToNumberOfLogs)
    st = []
    for strategy in strategies:
        st.extend([strategy] * (len(originalTL[0]) * numberOfLogs[-1] * 2))
    originalLogOrNotList = []
    for _ in strategies:
        ogORnotList = []
        ogORnotList.extend(['Average Trace Length Distribution of ' + str(numberOfLogs[-1]) + ' Play-Outs'] * (len(originalTL[0])) * numberOfLogs[-1])
        ogORnotList.extend(['Trace Length Distribution of the Original ' + logname + ' Log'] * (len(originalTL[0])) * numberOfLogs[-1])
        originalLogOrNotList.extend(ogORnotList)
    tupels = list(zip(traceLengths, st, originalLogOrNotList))
    return pd.DataFrame(tupels, columns=['Trace Length', 'Play-Out Strategy', 'Trace Length Distribution'])


def transfromTraceLengthsToDataframesHistograms(originalTL: list[list], generatedTLs: list[list], numberOfLogs: list,
                                      strategies: list, logname):
    originalTLscaledToNumberOfLogs = []
    weights = []
    weights.extend([1/numberOfLogs[-1]] * len(strategies) * len(originalTL[0]) * numberOfLogs[-1] * 2 )

    for i in range(numberOfLogs[-1]):
        originalTLscaledToNumberOfLogs.extend(originalTL[0])
    traceLengths = []
    for logs in generatedTLs:
        traceLengths.extend(originalTLscaledToNumberOfLogs)
        traceLengths.extend(logs)
    st = []
    for strategy in strategies:
        if strategy == "Quantifying the Re-identification\nRisk in Published Process Models":
            strategy = "Quantifying the\nRe-identification Risk in Published\nProcess Models"
        if strategy == "D with Variance 0.5":
            strategy = "D with\nVariance 0.5"
        if strategy == "D with Variance 1":
            strategy = "D with\nVariance 1"
        if strategy == "D with Variance 3":
            strategy = "D with\nVariance 3"
        if strategy == "D with Variance 5":
            strategy = "D with\nVariance 5"
        st.extend(['Play-Out Strategy ' + strategy] * (len(originalTL[0]) * numberOfLogs[-1] * 2))
    originalLogOrNotList = []
    ''' 
    if logname == "BPIC 2013 Closed Problems":
        name = "BPIC 2013 Closed\nProblems"
    elif logname == "BPIC 2015 Municipality 1":
        name = "BPIC 2015 Municipality 1\n"
    else:
        name = logname
    '''
    for _ in strategies:
        ogORnotList = []
        ogORnotList.extend(['Trace Length Distribution of the Original ' + logname + ' Log'] * (len(originalTL[0])) * numberOfLogs[-1])
        ogORnotList.extend(['Average Trace Length Distribution of ' + str(numberOfLogs[-1]) + ' Play-Outs'] * (len(originalTL[0])) * numberOfLogs[-1])
        originalLogOrNotList.extend(ogORnotList)
    tupels = list(zip(traceLengths, st, originalLogOrNotList, weights))
    return pd.DataFrame(tupels, columns=['Trace Length', 'Play-Out Strategy', 'Trace Length Distribution', 'Weights'])



def transfromOneLogTraceLengthsToDataframes(originalTL: list[list], generatedTLs: list[list],
                                      strategies: list, logname):
    traceLengths = []
    traceLengths.extend(originalTL[0])
    for logs in generatedTLs:
        traceLengths.extend(logs[:numberOfCasesInOriginalLog])
    st = []
    st.extend(["Original " + logname + " Log"] * numberOfCasesInOriginalLog)
    for strategy in strategies:
        st.extend([strategy] * numberOfCasesInOriginalLog)
    tupels = list(zip(traceLengths, st))
    return pd.DataFrame(tupels, columns=['Trace Length', 'Event Log'])


def transformListListToDataframe(listList: [[]], strategies: [], yname):
    flatList = []
    for l in listList:
        flatList.extend(l)
    st = []
    for strategy in strategies:
        st.extend([strategy] * (len(listList[0])))
    return pd.DataFrame(data=list(zip(flatList, st)), columns=[yname, 'Play-Out Strategy'])

def getAvgHistoOverlap(traceLengthsOG: [[]], traceLengthsListStrategies: [[]], numberOfLogs: []):
    lengthsOG = list(set(traceLengthsOG[0]))
    avgHistoOverlap = []
    for strategyList in traceLengthsListStrategies:
        count = 0
        for length in lengthsOG:
            stCount = strategyList.count(length)
            ogCount = traceLengthsOG[0].count(length) * numberOfLogs[-1]
            diff = stCount - ogCount
            if diff <= 0:
                count += stCount
            else: count += ogCount
        avgHistoOverlap.append(count/(len(traceLengthsOG[0])*numberOfLogs[-1]))
    return avgHistoOverlap


def addHistoIntersection(data, **kws):
    ax = plt.gca()
    ax.text(.8,.8,"Test 1234", transform = ax.transAxes)
    n = 10
    ax.text(4,4, f"$$HI = {n}$$")




abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#BPIC 2015 Municipality 1
#BPIC 2017
#Sepsis Cases
#BPIC 2013 Closed Problems

logName = "L"
print(logName)
citationMA = "Quantifying the Re-identification\nRisk in Published Process Models"
varianceList = [0.5, 1, 3, 5]
log = xes_importer.apply(logName + '.xes')
processTree = pm4py.discover_process_tree_inductive(log)
net, initial_marking, final_marking = inductive_miner.apply(log)

#processTree = wf_net_converter.apply(net, initial_marking, final_marking)
processTreeFreq = GenerationTree(processTree)
processTreeFreq = eventFreqPT(processTreeFreq, log)
numberOfCasesInOriginalLog = processTreeFreq.eventFreq

numberOfLogs = [100]
strategies = ["A", "B", "C"]
for variance in varianceList:
    strategies.append("D with Variance " + str(variance))
strategies.append(citationMA)  # "PM4PY basic playout of a Petri net", "PM4PY basic playout of a process tree", "PM4PY stochastic playout of a Petri net"]
#strategies.append("PM4Py's Stochastic Play-Out\nof a Petri Net")
numberOfTraceVariantListStrategies = list()
numberOfTraceLengthsListStrategies = list()
multiSetIntersectionSizeListStrategies = list()
emdListStrategies = list()
varianceCounter = 0
maxTraceLength = getLengthOfLongestTrace(log)

for strategy in strategies:
    generatedLogList, generatedEventLogList = getLogs(processTreeFreq, numberOfLogs[-1], numberOfCasesInOriginalLog,
                                                      strategy, initial_marking, final_marking, log, varianceList[varianceCounter], maxTraceLength=maxTraceLength)
    originalLogList = list()
    if "D with Variance" in strategy and varianceCounter != len(varianceList)-1:
        varianceCounter += 1
    originalLogList.append(transformLogToTraceStringList(log))

    # Trace Variants

    numberOfTraceVariantList = getNumberOfTraceVariantsList(generatedLogList)
    numberOfTraceVariantListStrategies.append(numberOfTraceVariantList)

    for i in numberOfLogs:
        print("Average Number of Trace Variants of " + str(i) + " generated Logs is: " + str(
            getAvgOfList(numberOfTraceVariantList, i)))
        print("Median Number of Trace Variants of " + str(i) + " generated Logs is: " + str(
            getMedianOfList(numberOfTraceVariantList, i)))
        print("Maximum Number of Trace Variants of " + str(i) + " generated Logs is: " + str(
            getMaxOfList(numberOfTraceVariantList, i)))
        print("Minimum Number of Trace Variants of " + str(i) + " generated Logs is: " + str(
            getMinOfList(numberOfTraceVariantList, i)))

    print("Number of Trace Variants of the original Log is: " + str(
        getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1)))

    # Trace Lengths

    numberOfTraceLengthsList = getTraceLengthsList(generatedLogList)
    numberOfTraceLengthsListStrategies.append(list(itertools.chain.from_iterable(numberOfTraceLengthsList)))

    for i in numberOfLogs:
        print("Average Length of a Trace in " + str(i) + " generated Logs is: " + str(
            getAverageFromListList(numberOfTraceLengthsList, i)))
        print("Average Median of Length of a Trace in " + str(i) + " generated Logs is: " + str(
            getAverageMediansFromListList(numberOfTraceLengthsList, i)))
        print("Maximum Length of a Trace in " + str(i) + " generated Logs is: " + str(
            getMaxFromListList(numberOfTraceLengthsList, i)))
        print("Minimum Length of a Trace in " + str(i) + " generated Logs is: " + str(
            getMinOfListList(numberOfTraceLengthsList, i)))

    numberOfTraceLengthsListOriginalLog = getTraceLengthsList(originalLogList)

    print("Average Length of a Trace in the original Logs is: " + str(
        getAverageFromListList(numberOfTraceLengthsListOriginalLog, 1)))
    print("Median Length of a Trace in the original Logs is: " + str(
        getAverageMediansFromListList(numberOfTraceLengthsListOriginalLog, 1)))
    print("Maximum Length of a Trace in the original Logs is: " + str(
        getMaxFromListList(numberOfTraceLengthsListOriginalLog, 1)))
    print("Minimum Length of a Trace in the original Logs is: " + str(
        getMinOfListList(numberOfTraceLengthsListOriginalLog, 1)))

    # Intersection with multi sets

    multiSetIntersectionSizeList = getMultiSetIntersection(generatedLogList, originalLogList[0])
    multiSetIntersectionSizeListStrategies.append(multiSetIntersectionSizeList)

    for i in numberOfLogs:
        print("Average Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getAvgOfList(multiSetIntersectionSizeList, i)))
        print("Maximum Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getMaxOfList(multiSetIntersectionSizeList, i)))
        print("Minimum Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getMinOfList(multiSetIntersectionSizeList, i)))

    #EMD
    '''
    emdList = getEMD(generatedEventLogList, log)
    emdListStrategies.append(emdList)

    for i in numberOfLogs:
        print("Average EMD of the original Log and " + str(i) + " generated Logs is: " + str(getAvgOfList(emdList, i)))
        print("Maximum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMaxOfList(emdList, i)))
        print("Minimum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMinOfList(emdList, i)))
    '''

#########################################
# Violin Plot for Number of Trace Variants
#########################################
#'''
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig.set_size_inches(10, 5.5)
#plt.gcf().subplots_adjust(bottom=0.26)
plt.grid(axis='both')
ax.set_ylabel('Number of Trace Variants')
set_axis_style(ax, strategies, variance)
ax.set_xlabel('Play-Out Strategy', loc='left', labelpad=5)

plt.axhline(y=getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1), color=(0.45, 0.71, 0.63),
            dashes=(4, 10), linestyle='--', linewidth=1,
            label='Number of Trace Variants in the Original ' + logName + ' Log.')
#' that has ' + str(len(log)) + ' Traces.')
ax = sns.violinplot(
    data=transformListListToDataframe(numberOfTraceVariantListStrategies, strategies, "Number of Trace Variants"),
    x="Play-Out Strategy", y="Number of Trace Variants", inner="stick", bw=0.5, scale="area", linewidth=0.7,
    color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))
trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(1.025, getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1),
        "{:.0f}".format(getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1)),
        color=(0.4, 0.7607843137254902, 0.6470588235294118), transform=trans,
        ha="left", va="center")
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=2)
plt.tight_layout()
plt.savefig("pdf/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".png", format="png",
            dpi=3000, transparent=True)
plt.savefig("svg/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()
#'''
#########################################
# Violin Plot for Number Of Trace Lengths
#########################################

""" 
pdList = transfromTraceLengthsToDataframes(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies,
                                           numberOfLogs, strategies, logName)


if logName == 'BPIC 2017':
    maxLength = 100
elif logName == 'Sepsis Cases':
    maxLength = 40
elif logName == 'BPIC 2015 Municipality 1':
    maxLength = 110
else:
    maxLength = 20

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 17})

rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes

fig = plt.figure()
plt.grid(axis='y')
#pdList.drop(pdList.index[pdList['Trace Lengths'] > maxLength], inplace=True)
set_axis_style(ax, strategies, variance)

if logName == "BPIC 2013 Closed Problems":
    fig.set_size_inches(13, 6)
else:
    fig.set_size_inches(13, 8.5)

if logName == "BPIC 2013 Closed Problems":
    ax = sns.violinplot(x='Play-Out Strategy', y='Trace Length', hue='Trace Length Distribution',
                        data=pdList, palette=[(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                                              (0.4, 0.7607843137254902, 0.6470588235294118)], split=True,
                        scale="area", width=0.95, cut = 0, gridsize = numberOfCasesInOriginalLog, inner = None,
                        scale_hue=False, bw = 0.19, linewidth = 0.3)
elif logName == "BPIC 2017":
    ax = sns.violinplot(x='Play-Out Strategy', y='Trace Length', hue='Trace Length Distribution',
                    data=pdList, palette=[(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                                          (0.4, 0.7607843137254902, 0.6470588235294118)], split=True,
                    scale="area", width=0.95, inner = None, bw = 0.025, cut = 0, gridsize = numberOfCasesInOriginalLog,
                    scale_hue=False, linewidth = 0.3)


else:
    ax = sns.violinplot(x='Play-Out Strategy', y='Trace Length', hue='Trace Length Distribution',
                    data=pdList, palette=[(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                                          (0.4, 0.7607843137254902, 0.6470588235294118)], split=True,
                    scale="area", width=0.95, inner = None, bw = 0.08, cut = 0, gridsize = numberOfCasesInOriginalLog,
                    scale_hue=False, linewidth = 0.3)


ax.set(ylim = (0, maxLength))

plt.tight_layout()

plt.savefig("pdf/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=3000,
            transparent=True)
plt.savefig("svg/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()
"""

#########################################
# Scatter Plot Length above maxLength
#########################################
'''
strategyAvgLengths = list()
strategyAvgCounts = list()

for strategy in strategies:
    strategyAvgLength = len(pdList.loc[(pdList['Trace Lengths'] > maxLength) & (
                pdList['Play-Out Strategy'] == strategy) & (pdList['Trace Lengths in the:'] == "Average Trace Length Distribution of " + str(numberOfLogs[-1]) + " Play-Outs")].index) / numberOfLogs[-1]
    strategyAvgCount = pdList.loc[(pdList['Trace Lengths'] > maxLength) & (
            pdList['Play-Out Strategy'] == strategy) & (pdList['Trace Lengths in the:'] == "Average Trace Length Distribution of " + str(numberOfLogs[-1]) + " Play-Outs")]["Trace Lengths"].mean()
    strategyAvgLengths.append(strategyAvgLength)
    strategyAvgCounts.append(strategyAvgCount)
    print(strategy + ' has an Average of ' + str(strategyAvgLength) + ' traces with a length greater than ' + str(
        maxLength))
    print("Those traces have an average length of " + str(strategyAvgCount))

tupels = list(zip(strategyAvgLengths, strategyAvgCounts, strategies))

scatterLengths = pd.DataFrame(tupels, columns=['strategyAvgLengths', 'strategyAvgCounts',  'strategies'])

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig.set_size_inches(10, 5.5)
plt.gcf().subplots_adjust(bottom=0.35)
plt.grid(axis='y')
# ax.set_title('Violin plots and the means of the intersection size from ' + str(numberOfLogs[-1]) + ' simulated logs generated by different strategies with the ' + logName + ' log.')
ax.set_ylabel('Average Trace Length of Traces with Length \n above ' + str(maxLength) + ' for the ' + logName + ' log')
ax.set_xlabel('Average Count of Traces with Length \n above ' + str(maxLength) + ' for the ' + logName + ' log', loc='left')
#set_axis_style(ax, strategies, variance)
#ax = sns.violinplot(data=transformListListToDataframe(multiSetIntersectionSizeListStrategies, strategies,
#                                                      'Multi set intersection size w. ' + logName), x="Strategy",
 #                   y='Multi set intersection size w. ' + logName, inner="stick", scale="area", bw=0.2, linewidth=1,
  #                  color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))

ax = sns.scatterplot(data = scatterLengths, x = 'strategyAvgLengths', y='strategyAvgCounts', hue = 'strategies', style = 'strategies')
plt.gcf().subplots_adjust(bottom=0.35)
plt.legend(ncol = 3)
#plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.8), loc="lower left",
#           mode="expand", borderaxespad=0, ncol=1)
plt.tight_layout()

plt.savefig("pdf/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=3000,
            transparent=True)
plt.savefig("svg/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()
'''

#########################################
# Histogram of Trace Lengths
#########################################


if logName == 'BPIC 2017':
    maxLength = 100
elif logName == 'Sepsis Cases':
    maxLength = 60
elif logName == 'BPIC 2015 Municipality 1':
    maxLength = 90
elif logName == 'BPIC 2013 Closed Problems':
    maxLength = 30
else:
    maxLength = maxTraceLength + 20

#pdList = transfromOneLogTraceLengthsToDataframes(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies,
#                                            strategies, logName)

pdList = transfromTraceLengthsToDataframesHistograms(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies,
                                           numberOfLogs, strategies, logName)
avgHOL = getAvgHistoOverlap(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies, numberOfLogs)

print("------------------------------------------------------------------------------------------------------------")
print("Avg Histo Overlap: " + str(avgHOL))

fig, ax = plt.subplots()  # a figure with a single Axes
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})

ax = sns.displot(
    pdList, x="Trace Length", col="Play-Out Strategy", hue = 'Trace Length Distribution', weights = 'Weights',
    binwidth = 1, col_wrap=4,
    aspect=1*0.5, facet_kws=dict(margin_titles=True, despine = False), kind="hist"
)

sns.move_legend(
    ax, "center", bbox_to_anchor=(.5, .95), shadow = True,
     ncol=2, title=None, frameon=True, fancybox = True,
)

""" 
for i in range(8):
    ax1 = ax.axes[i]
    nhi = float(f'{avgHOL[i]:.3f}')
    ax1.text(.95, .95, f"$$NHI = {nhi}$$", horizontalalignment='right',
            verticalalignment='top',
            transform=ax1.transAxes)
"""
ax1 = ax.axes[0]
ax1.text(1.2, 1.35, " 123", horizontalalignment='right',
        verticalalignment='top',
        transform=ax1.transAxes)


#ax.map_dataframe(addHistoIntersection(avgHOL))
#ax.add_legend()
#values_sns = [h.get_height() for h in ax.patches]
#bins_sns = [h.get_width() for h in ax.patches]
#plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
#           mode="expand", borderaxespad=0, ncol=2)
#plt.subplots_adjust(top = 3)
#ax._legend_out = False

''' 
sns.move_legend(
    ax, "center", bbox_to_anchor=(.833, .2),
     ncol=1, title=None, frameon=True, fancybox = True,
)
'''
#leg = ax.legend()


#ax.text(0.5,1, "Test 123123123123")
ax.set_titles('{col_name}')
ax.set(xlim = (0, maxLength))
#fig.set_size_inches(17.56, 10)
plt.tight_layout()
#plt.subplots_adjust(top=0.2)
Path("pdf/" + logName + "/Violin/Histogram").mkdir(parents=True, exist_ok=True)
Path("png/" + logName + "/Violin/Histogram").mkdir(parents=True, exist_ok=True)
Path("svg/" + logName + "/Violin/Histogram").mkdir(parents=True, exist_ok=True)
plt.savefig("pdf/" + logName + "/Violin/Histogram/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/Histogram/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=300,
            transparent=True)
plt.savefig("svg/" + logName + "/Violin/Histogram/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()


#########################################
# Violin Plot for Histogram Overlap Percentage
#########################################
''' 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig.set_size_inches(10, 5.5)

plt.gcf().subplots_adjust(bottom=0.35)
plt.grid(axis='y')
# ax.set_title('Violin plots and the means of the intersection size from ' + str(numberOfLogs[-1]) + ' simulated logs generated by different strategies with the ' + logName + ' log.')
ax.set_ylabel('P\n' + logName + ' Log')
ax.set_xlabel('Play-Out Strategy', loc='left')
set_axis_style(ax, strategies, variance)
ax = sns.violinplot(data=transformListListToDataframe(multiSetIntersectionSizeListStrategies, strategies,
                                                      'Multiset Intersection Size w. t. Original\n' + logName + ' Log'), x="Play-Out Strategy",
                    y='Multiset Intersection Size w. t. Original\n' + logName + ' Log', inner="stick", scale="area", bw=0.5, linewidth=1,
                    color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))
plt.tight_layout()
Path("pdf/" + logName + "/Violin/Overlap").mkdir(parents=True, exist_ok=True)
Path("png/" + logName + "/Violin/Overlap").mkdir(parents=True, exist_ok=True)
Path("svg/" + logName + "/Violin/Overlap").mkdir(parents=True, exist_ok=True)
plt.savefig("pdf/" + logName + "/Violin/Overlap/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/Overlap/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=3000,
            transparent=True)
plt.savefig("svg/" + logName + "/Violin/Overlap/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()
'''


#########################################
# Violin Plot for Number Of Multi Set Intersection
#########################################


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig.set_size_inches(10, 5.5)

plt.gcf().subplots_adjust(bottom=0.35)
plt.grid(axis='y')
# ax.set_title('Violin plots and the means of the intersection size from ' + str(numberOfLogs[-1]) + ' simulated logs generated by different strategies with the ' + logName + ' log.')
ax.set_ylabel('Multiset Intersection Size with the Original\n' + logName + ' Log')
ax.set_xlabel('Play-Out Strategy', loc='left')
set_axis_style(ax, strategies, variance)
ax = sns.violinplot(data=transformListListToDataframe(multiSetIntersectionSizeListStrategies, strategies,
                                                      'Multiset Intersection Size with the Original\n' + logName + ' Log'), x="Play-Out Strategy",
                    y='Multiset Intersection Size with the Original\n' + logName + ' Log', inner="stick", scale="area", bw=0.5, linewidth=1,
                    color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))
plt.tight_layout()

plt.savefig("pdf/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=3000,
            transparent=True)
plt.savefig("svg/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".svg", format="svg",
            transparent=True)
plt.show()

#########################################
# Violin Plot for EMD
#########################################
""""
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig.set_size_inches(10, 5.5)
plt.gcf().subplots_adjust(bottom=0.35)
plt.grid(axis='y')
ax.set_ylabel('EMD with ' + logName + ' Log')
ax.set_xlabel('Play-Out Strategy', loc='left')
set_axis_style(ax, strategies, variance)
ax = sns.violinplot(data = transformListListToDataframe(emdListStrategies, strategies,"EMD with the Original\n" + logName + " Log"), x = "Play-Out Strategy", y = "EMD with the Original\n" + logName + " Log", inner="stick", bw = 0.5, scale = "area", color = (0.9882352941176471, 0.5529411764705883, 0.3843137254901961), linewidth=1,)
plt.tight_layout()

plt.savefig("pdf/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".pdf", format = "pdf", transparent=True)
plt.savefig("png/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".png", format = "png", dpi=3000, transparent=True)
plt.savefig("svg/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".svg", format = "svg", transparent=True)
plt.show()

"""