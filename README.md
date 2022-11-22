# reconstruction-thesis

# Usage

1. Set in evaluation.py in line 336 `logName` to the name of the xes file pf the log we want to import.

    Example. We want to import the "BPIC 2013 Closed Problems.xes" fiel:

    `logName = 'BPIC 2013 Closed Problems.xes'` 

2. Set in evaluation.py in line 337 the number of logs `numberOfLogs` to the number of reconstructed logs each play-out strategy should generate.

    Example. When each play-out strategy should generate should reconstruct 100 logs from a process tree we type:

    `numberOfLogs = 100` 

3. Run the evaluation.py file.
