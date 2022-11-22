import copy
import itertools
from collections import Counter as mset
from statistics import median

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.process_tree import semantics
from pm4py.statistics.variants.log import get as variants_module

from frequencyAnnotationOfProcessTree import eventFreqPT
from strategyC import generateLog as generateLogDistribution
from strategyDwithVarianceD import GenerationTree
from strategyDwithVarianceD import generateLog as generateLogFreq
from strategyInQuantification import generateLog as generateLogMA
from strategyA import generateLog as generateLogUniform


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


def getLogs(processTree, numberOfLogsToGenerate, numberOfCasesInOriginalLog, strategy, im, fm, originalLog, variance):
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
            log, eventlog = generateLogDistribution(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == ("C with Variance " + str(variance)):
            log, eventlog = generateLogFreq(processTreeCopy, numberOfCasesInOriginalLog, variance)
        elif strategy == citationMA:
            log, eventlog = generateLogMA(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "PM4PY basic playout of a Petri net":
            eventlog = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
                simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: numberOfCasesInOriginalLog})
            log = transformLogToTraceStringList(eventlog)
        elif strategy == "PM4PY basic playout of a process tree":
            eventlog = semantics.generate_log(processTreeCopy, no_traces=numberOfCasesInOriginalLog)
            log = transformLogToTraceStringList(eventlog)
        elif strategy == "PM4PY stochastic Play-Out of a Petri Net":
            eventlog = simulator.apply(net, im, fm, parameters={
                simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: originalLog,
                simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.NO_TRACES: numberOfCasesInOriginalLog},
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
    ax.set_xlabel('Strategy')


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
        ogORnotList.extend(['Simulated Logs'] * (len(originalTL[0])) * numberOfLogs[-1])
        ogORnotList.extend(['Original ' + logname + ' Log'] * (len(originalTL[0])) * numberOfLogs[-1])
        originalLogOrNotList.extend(ogORnotList)
    tupels = list(zip(traceLengths, st, originalLogOrNotList))
    return pd.DataFrame(tupels, columns=['Trace Lengths', 'Strategy', 'Trace Lengths in the:'])


def transformListListToDataframe(listList: [[]], strategies: [], yname):
    flatList = []
    for l in listList:
        flatList.extend(l)
    st = []
    for strategy in strategies:
        st.extend([strategy] * (len(listList[0])))
    return pd.DataFrame(data=list(zip(flatList, st)), columns=[yname, 'Strategy'])


logNames = ["Sepsis Cases", "Sepsis Cases"]
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)

figTraceVariants, axesTraceVariants = plt.subplots(nrows=len(logNames), sharex=True)
figTraceVariants.set_size_inches(10, 6.5 * len(logNames))

logNameCounter = 0
for logName in logNames:
    citationMA = "Quantifying the Re-identification Risk \nin Published Process Models"
    varianceList = [0.5, 1, 2, 3, 5, 10]
    log = xes_importer.apply(logName + '.xes')
    net, initial_marking, final_marking = inductive_miner.apply(log)

    processTree = wf_net_converter.apply(net, initial_marking, final_marking)
    processTreeFreq = GenerationTree(processTree)
    processTreeFreq = eventFreqPT(processTreeFreq, log)
    numberOfCasesInOriginalLog = processTreeFreq.eventFreq

    numberOfLogs = [2]
    strategies = ["A", "B"]
    for variance in varianceList:
        strategies.append("C with Variance " + str(variance))
    strategies.append(
        citationMA)  # "PM4PY basic playout of a Petri net", "PM4PY basic playout of a process tree", "PM4PY stochastic playout of a Petri net"]
    numberOfTraceVariantListStrategies = list()
    numberOfTraceLengthsListStrategies = list()
    multiSetIntersectionSizeListStrategies = list()
    emdListStrategies = list()
    varianceCounter = 0

    for strategy in strategies:
        generatedLogList, generatedEventLogList = getLogs(processTreeFreq, numberOfLogs[-1], numberOfCasesInOriginalLog,
                                                          strategy, initial_marking, final_marking, log,
                                                          varianceList[varianceCounter])
        originalLogList = list()
        if "C with Variance" in strategy and varianceCounter != len(varianceList) - 1:
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

        # EMD
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

    if logNameCounter == len(logNames) - 1:
        plt.gcf().subplots_adjust(bottom=0.35)
        plt.grid(axis='y')
        set_axis_style(axesTraceVariants[logNameCounter], strategies, variance)
        axesTraceVariants[logNameCounter].set_xlabel('Strategy', loc='left')

    axesTraceVariants[logNameCounter].set_ylabel('Number of Trace Variants')
    plt.axhline(y=getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1), color=(0.45, 0.71, 0.63),
                dashes=(4, 10), linestyle='--', linewidth=1,
                label='Number of Trace Variants in the original ' + logName + ' Log.')
    # ' that has ' + str(len(log)) + ' Traces.')
    axesTraceVariants[logNameCounter] = sns.violinplot(
        data=transformListListToDataframe(numberOfTraceVariantListStrategies, strategies, "Number of Trace Variants"),
        x="Strategy", y="Number of Trace Variants", inner="stick", bw=0.5, scale="area",
        color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))
    trans = transforms.blended_transform_factory(
        axesTraceVariants[logNameCounter].get_yticklabels()[0].get_transform(),
        axesTraceVariants[logNameCounter].transData)
    axesTraceVariants[logNameCounter].text(1.025, getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1),
                                           "{:.0f}".format(
                                               getMaxOfList(getNumberOfTraceVariantsList(list(originalLogList)), 1)),
                                           color=(0.4, 0.7607843137254902, 0.6470588235294118), transform=trans,
                                           ha="left", va="center")
    if logNameCounter == len(logNames) - 1:
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                   mode="expand", borderaxespad=0, ncol=2)
        plt.savefig("pdf/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
                    transparent=True)
        plt.savefig("png/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".png", format="png",
                    dpi=3000, transparent=True)
        plt.savefig("svg/" + logName + "/Violin/NumberOfTraceVariants/" + str(numberOfLogs[-1]) + ".svg", format="svg",
                    transparent=True)
        plt.show()

    #########################################
    # Violin Plot for Number Of Trace Lengths
    #########################################
    ''' 
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    rc('text', usetex=True)
    fig = plt.figure()  # an empty figure with no Axes

    fig = plt.figure()
    pdList = transfromTraceLengthsToDataframes(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies,
                                               numberOfLogs, strategies, logName)
    plt.grid(axis='y')

    if logName == 'BPIC 2017':
        maxLength = 200
    elif logName == 'Sepsis Cases':
        maxLength = 40
    elif logName == 'BPIC 2015 Municipality 1':
        maxLength = 100
    else:
        maxLength = 75
    for strategy in strategies:
        print(strategy + ' has an Average of ' + str(len(pdList.loc[(pdList['Trace Lengths'] > maxLength) & (
                pdList['Strategy'] == strategy) & (pdList['Trace Lengths in the:'] == "Simulated Logs")].index) /
                                                     numberOfLogs[-1]) + ' traces with a length greater than ' + str(
            maxLength))
        print("Those traces have an average length of " + str(pdList.loc[(pdList['Trace Lengths'] > maxLength) & (
                pdList['Strategy'] == strategy) & (pdList['Trace Lengths in the:'] == "Simulated Logs")][
                                                                  "Trace Lengths"].mean()))

    # pdList.drop(pdList.index[pdList['Trace Lengths'] > maxLength], inplace=True)
    fig.set_size_inches(10, 6.5)
    # 12, 7
    set_axis_style(ax, strategies, variance)

    ax = sns.violinplot(x='Strategy', y='Trace Lengths', hue='Trace Lengths in the:',
                        data=pdList, palette=[(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                                              (0.4, 0.7607843137254902, 0.6470588235294118)], split=True,
                        linewidth=0.01,
                        scale="area", width=0.95, bw=0.07,
                        scale_hue=False)
    ax.set(ylim=(0, maxLength))
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=2)

    plt.savefig("pdf/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
                transparent=True)
    plt.savefig("png/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".png", format="png",
                dpi=3000,
                transparent=True)
    plt.savefig("svg/" + logName + "/Violin/NumberOfTraceLengths/" + str(numberOfLogs[-1]) + ".svg", format="svg",
                transparent=True)
    plt.show()

    #########################################
    # Violin Plot for Number Of Multi Set Intersection
    #########################################

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    rc('text', usetex=True)
    fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots()  # a figure with a single Axes
    fig.set_size_inches(10, 6.5)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.grid(axis='y')
    # ax.set_title('Violin plots and the means of the intersection size from ' + str(numberOfLogs[-1]) + ' simulated logs generated by different strategies with the ' + logName + ' log.')
    ax.set_ylabel('Multiset Intersection Size with ' + logName)
    ax.set_xlabel('Strategy', loc='left')
    set_axis_style(ax, strategies, variance)
    ax = sns.violinplot(data=transformListListToDataframe(multiSetIntersectionSizeListStrategies, strategies,
                                                          'Multi set intersection size w. ' + logName), x="Strategy",
                        y='Multi set intersection size w. ' + logName, inner="stick", scale="area", bw=0.2,
                        color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961))

    plt.savefig("pdf/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
                transparent=True)
    plt.savefig("png/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=3000,
                transparent=True)
    plt.savefig("svg/" + logName + "/Violin/IntersectionSize/" + str(numberOfLogs[-1]) + ".svg", format="svg",
                transparent=True)
    plt.show()
    '''
    #########################################
    # Violin Plot for EMD
    #########################################
    ''' 
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    rc('text', usetex=True)
    fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots()  # a figure with a single Axes
    fig.set_size_inches(10, 6.25)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.grid(axis='y')
    ax.set_ylabel('EMD with ' + logName + ' Log')
    ax.set_xlabel('Strategy', loc='left')
    set_axis_style(ax, strategies, variance)
    ax = sns.violinplot(data = transformListListToDataframe(emdListStrategies, strategies,"EMD w. " + logName + " log"), x = "Strategy", y = "EMD w. " + logName + " log", inner="stick", bw = 0.5, scale = "area", color = (0.9882352941176471, 0.5529411764705883, 0.3843137254901961))
    plt.savefig("pdf/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".pdf", format = "pdf", transparent=True)
    plt.savefig("png/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".png", format = "png", dpi=3000, transparent=True)
    plt.savefig("svg/" + logName + "/Violin/EMD/" + str(numberOfLogs[-1]) + ".svg", format = "svg", transparent=True)
    plt.show()

    
    '''
    logNameCounter += 1
