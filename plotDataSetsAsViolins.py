import pm4py
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
import pandas as pd



def getTraceLengthsList(logList):
    logTraceLengths = list()
    for log in logList:
        traceLengths = list()
        for trace in log:
            traceLengths.append(len(trace))
        logTraceLengths.append(traceLengths)
    return logTraceLengths

def transformListListToDataframe(flatList):
    st = []
    st.extend([""] * (len(flatList)))
    return pd.DataFrame(data=list(zip(flatList, st)), columns=['Trace Length', logName])
#BPIC 2015 Municipality 1
#BPIC 2013 Closed Problems
logName = "BPIC 2017"
log = pm4py.read_xes(logName + '.xes')
logName = logName + " Log"
loglist = []
loglist.append(log)
numberOfTraceLengthsList = getTraceLengthsList(loglist)
pdList = transformListListToDataframe(numberOfTraceLengthsList[0])
numberOfCasesInOriginalLog = len(numberOfTraceLengthsList[0])
''' 
if logName == 'BPIC 2017 Log':
    maxLength = 100
elif logName == 'Sepsis Cases Log':
    maxLength = 180
elif logName == 'BPIC 2015 Municipality 1 Log':
    maxLength = 110
else:
    maxLength = 20
'''
maxLength = max(numberOfTraceLengthsList[0])

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 17})
rc('text', usetex=True)
fig = plt.figure()  # an empty figure with no Axes

fig = plt.figure()
plt.grid(axis='y')
ax = sns.violinplot()
fig.set_size_inches(4, 8.5)

if logName == "BPIC 2013 Closed Problems Log":
    ax = sns.violinplot(x = logName ,y='Trace Length',
                        data=pdList, palette=[
                                              (0.4, 0.7607843137254902, 0.6470588235294118)],
                        scale="area", width=0.95, cut = 0, gridsize = numberOfCasesInOriginalLog, inner = None,
                        bw = 0.13, linewidth = 0.5)
elif logName == "BPIC 2017 Log":
    ax = sns.violinplot(x = logName ,y='Trace Length',
                    data=pdList, palette=[
                                          (0.4, 0.7607843137254902, 0.6470588235294118)],
                    scale="area", width=0.95, inner = None, bw = 0.03, cut = 0, gridsize = numberOfCasesInOriginalLog,
                     linewidth = 0.5)
elif logName == "BPIC 2015 Municipality 1 Log":
    ax = sns.violinplot(x = logName ,y='Trace Length',
                    data=pdList, palette=[
                                          (0.4, 0.7607843137254902, 0.6470588235294118)],
                    scale="area", width=0.95, inner = None, bw = 0.06, cut = 0, gridsize = numberOfCasesInOriginalLog,
                         linewidth = 0.5)
else:
    ax = sns.violinplot(x = logName ,y='Trace Length',
                    data=pdList, palette=[
                                          (0.4, 0.7607843137254902, 0.6470588235294118)],
                    scale="area", width=0.95, inner = None, bw = 0.06, cut = 0, gridsize = numberOfCasesInOriginalLog,
                         linewidth = 0.5)

ax.set(ylim = (0, maxLength))

plt.tight_layout()
plt.savefig("dis/" + logName + ".pdf", format="pdf",
            transparent=True)
plt.show()

