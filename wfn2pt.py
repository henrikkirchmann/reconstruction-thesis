import copy

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.algo.conformance.alignments.process_tree.util import search_graph_pt_frequency_annotation
from generateLogFreq import generateLog, GenerationTree
import time

from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from workFlowNet2ProcessTree.isSeqRandom import freqOfCheckTicketSecondPositionInTrace
from workFlowNet2ProcessTree.isLoopRandom import freqOfLoop
from workFlowNet2ProcessTree.xorFreq import freqOfXOR
from workFlowNet2ProcessTree.eventFrequencyPT import eventFreqPT


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


log = xes_importer.apply('/Logs/Sepsis Cases.xes')  # Importiere die xes-Datei des Logs

net, initial_marking, final_marking = inductive_miner.apply(log)
''' 
parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "svg"}
gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters, variant=pn_visualizer.Variants.FREQUENCY, log=log)
pn_visualizer.save(gviz, "inductive_frequency.svg")
'''





tree = wf_net_converter.apply(net, initial_marking, final_marking)
''' 
aligned_traces = pm4py.conformance_diagnostics_alignments(log, tree)
tree = search_graph_pt_frequency_annotation.apply(tree, aligned_traces)
gviz = pt_visualizer.apply(tree, parameters={"format": "svg"}, variant=pt_visualizer.Variants.FREQUENCY_ANNOTATION)
pt_visualizer.view(gviz)
'''
gviz = pt_visualizer.apply(tree, parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"})
pt_visualizer.view(gviz)



''' 
start = time.time()
aligned_traces = pm4py.conformance_diagnostics_alignments(log, tree)
treeA = search_graph_pt_frequency_annotation.apply(tree, aligned_traces)
end = time.time()
print(end - start)
'''


ls = 100
logs = list()
pt = GenerationTree(tree)
pt = eventFreqPT(pt, log)
number_of_cases = pt.eventFreq #Number of Cases in the wf net
for i in range(ls):
    ptc = copy.deepcopy(pt)
    logs.append(generateLog(ptc, number_of_cases))

for i in range(ls-1):
    logs[0] = [x for x in logs[0] if x in logs[i+1]]
    print("Schnittmenge nach " + str(i+1) + "generierten Logs: " + str(len(logs[0])))
log_list = transformLogToTraceStringList(log)





''' 
c = 0
t = 0
for trace in log:
    stringTrace = transformTraceToString(trace)
    s = freqOfXOR(stringTrace)
    print(stringTrace)
    if s == "t":
        t += 1
    if s == "c":
        c += 1
print(t)
print(c)
print(str(t+c))
'''
