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

def transformListListToDataframe(lenList):
    st = []
    i = 0
    for logName in logNameList:
        st.extend([logName] * len(logList[i]))
        i += 1
    ll =[]
    for lengths in lenList:
        ll.extend(lengths)
    return pd.DataFrame(data=list(zip(ll, st)), columns=['Trace Length', "Event Log"])


#BPIC 2017
#BPIC 2015 Municipality 1
#BPIC 2013 Closed Problems
#Sepsis Cases
logNameList = ["BPIC 2017", "BPIC 2015 Municipality 1", "BPIC 2013 Closed Problems", "Sepsis Cases"]
#logName = "Sepsis Cases"
logList = []
for logName in logNameList:
    log = pm4py.read_xes("/vol/fob-vol4/mi17/kirchmah/PycharmProjects/BA/" + logName + ".xes")
    logName = logName + " Log"
    logList.append(log)
numberOfTraceLengthsList = getTraceLengthsList(logList)
pdList = transformListListToDataframe(numberOfTraceLengthsList)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
rc('text', usetex=True)

fig, ax = plt.subplots()
plt.grid(axis='y')
#fig.set_size_inches(13, 8.5)
print(pdList)
ax = sns.displot(
    pdList, x="Trace Length", col="Event Log", aspect=0.37*4.25,
    binwidth = 1, col_wrap=2,
  facet_kws=dict(margin_titles=True, sharex = False, sharey = False), kind="hist"
)


ax.set_titles('{col_name}')

plt.tight_layout()
plt.savefig("/vol/fob-vol4/mi17/kirchmah/PycharmProjects/BA/dis/" + logName + ".pdf", format="pdf",
            transparent=True)
plt.show()
